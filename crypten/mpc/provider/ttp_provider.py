#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from functools import reduce

import crypten
import crypten.communicator as comm
import torch
import torch.distributed as dist
from crypten.common.rng import generate_kbit_random_tensor, generate_random_ring_element
from crypten.common.util import count_wraps, torch_stack
from crypten.encoder import FixedPointEncoder
from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor


TTP_FUNCTIONS = ["additive", "square", "binary", "wraps", "B2A", "rand"]


class TrustedThirdParty:
    NAME = "TTP"

    @staticmethod
    def generate_additive_triple(size0, size1, op, device=None, *args, **kwargs):
        """Generate multiplicative triples of given sizes"""
        generator = TTPClient.get().generator

        a = generate_random_ring_element(size0, generator=generator, device=device)
        b = generate_random_ring_element(size1, generator=generator, device=device)
        if comm.get().get_rank() == 0:
            # Request c from TTP
            c = TTPClient.get().ttp_request(
                "additive", size0, size1, op, *args, **kwargs
            )
        else:
            # TODO: Compute size without executing computation
            #c_size = getattr(torch, op)(a, b, *args, **kwargs).size()
            c_size = comm.get().recv_obj(comm.get().get_ttp_rank(), comm.get().ttp_group)
            c = generate_random_ring_element(c_size, generator=generator, device=device)

        a = ArithmeticSharedTensor.from_shares(a, precision=0)
        b = ArithmeticSharedTensor.from_shares(b, precision=0)
        c = ArithmeticSharedTensor.from_shares(c, precision=0)
        return a, b, c

    @staticmethod
    def square(size, device=None):
        """Generate square double of given size"""
        generator = TTPClient.get().generator

        r = generate_random_ring_element(size, generator=generator, device=device)
        if comm.get().get_rank() == 0:
            # Request r2 from TTP
            r2 = TTPClient.get().ttp_request("square", size)
        else:
            r2 = generate_random_ring_element(size, generator=generator, device=device)

        r = ArithmeticSharedTensor.from_shares(r, precision=0)
        r2 = ArithmeticSharedTensor.from_shares(r2, precision=0)
        return r, r2

    @staticmethod
    def generate_binary_triple(size0, size1, device=None):
        """Generate binary triples of given size"""
        generator = TTPClient.get().generator

        a = generate_kbit_random_tensor(size0, generator=generator, device=device)
        b = generate_kbit_random_tensor(size1, generator=generator, device=device)

        if comm.get().get_rank() == 0:
            # Request c from TTP
            c = TTPClient.get().ttp_request("binary", size0, size1)
        else:
            size2 = torch.broadcast_tensors(a, b)[0].size()
            c = generate_kbit_random_tensor(size2, generator=generator, device=device)

        # Stack to vectorize scatter function
        a = BinarySharedTensor.from_shares(a)
        b = BinarySharedTensor.from_shares(b)
        c = BinarySharedTensor.from_shares(c)
        return a, b, c

    @staticmethod
    def wrap_rng(size, device=None):
        """Generate random shared tensor of given size and sharing of its wraps"""
        generator = TTPClient.get().generator

        r = generate_random_ring_element(size, generator=generator, device=device)
        if comm.get().get_rank() == 0:
            # Request theta_r from TTP
            theta_r = TTPClient.get().ttp_request("wraps", size)
        else:
            theta_r = generate_random_ring_element(
                size, generator=generator, device=device
            )

        r = ArithmeticSharedTensor.from_shares(r, precision=0)
        theta_r = ArithmeticSharedTensor.from_shares(theta_r, precision=0)
        return r, theta_r

    @staticmethod
    def B2A_rng(size, device=None):
        """Generate random bit tensor as arithmetic and binary shared tensors"""
        generator = TTPClient.get().generator

        # generate random bit
        rB = generate_kbit_random_tensor(
            size, bitlength=1, generator=generator, device=device
        )

        if comm.get().get_rank() == 0:
            # Request rA from TTP
            rA = TTPClient.get().ttp_request("B2A", size)
        else:
            rA = generate_random_ring_element(size, generator=generator, device=device)

        rA = ArithmeticSharedTensor.from_shares(rA, precision=0)
        rB = BinarySharedTensor.from_shares(rB)
        return rA, rB

    @staticmethod
    def rand(*sizes, encoder=None, device=None):
        """Generate random ArithmeticSharedTensor uniform on [0, 1]"""
        generator = TTPClient.get().generator

        if isinstance(sizes, torch.Size):
            sizes = tuple(sizes)

        if isinstance(sizes[0], torch.Size):
            sizes = tuple(sizes[0])

        if comm.get().get_rank() == 0:
            # Request samples from TTP
            samples = TTPClient.get().ttp_request("rand", *sizes, encoder=encoder)
        else:
            samples = generate_random_ring_element(
                sizes, generator=generator, device=device
            )
        return ArithmeticSharedTensor.from_shares(samples)

    @staticmethod
    def _init():
        TTPClient._init()

    @staticmethod
    def uninit():
        TTPClient.uninit()


class TTPClient:
    __instance = None

    class __TTPClient:
        """Singleton class"""

        def __init__(self, device=None):
            # Initialize connection
            self.ttp_group = comm.get().ttp_group
            self.comm_group = comm.get().ttp_comm_group
            self.device = "cuda"
            self._setup_generators()
            logging.info(f"TTPClient {comm.get().get_rank()} initialized")

        def _setup_generators(self):
            seed = torch.empty(size=(), dtype=torch.long)
            dist.irecv(
                tensor=seed, src=comm.get().get_ttp_rank(), group=self.ttp_group
            ).wait()
            dist.barrier(group=self.ttp_group)

            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(seed.item())

        def ttp_request(self, func_name, *args, **kwargs):
            assert (
                comm.get().get_rank() == 0
            ), "Only party 0 communicates with the TTPServer"
            message = {"function": func_name, "args": args, "kwargs": kwargs}
            ttp_rank = comm.get().get_ttp_rank()

            comm.get().send_obj(message, ttp_rank, self.ttp_group)

            size = comm.get().recv_obj(ttp_rank, self.ttp_group)
            result = torch.empty(size, dtype=torch.long, device=self.device)
            comm.get().broadcast(result, ttp_rank, self.comm_group)

            return result

    @staticmethod
    def _init():
        """Initializes a Trusted Third Party client that sends requests"""
        if TTPClient.__instance is None:
            TTPClient.__instance = TTPClient.__TTPClient()

    @staticmethod
    def uninit():
        """Uninitializes a Trusted Third Party client"""
        del TTPClient.__instance
        TTPClient.__instance = None

    @staticmethod
    def get():
        """Returns the instance of the TTPClient"""
        if TTPClient.__instance is None:
            raise RuntimeError("TTPClient is not initialized")

        return TTPClient.__instance


class TTPServer:
    TERMINATE = -1

    def __init__(self, device=None):
        """Initializes a Trusted Third Party server that receives requests"""
        # Initialize connection
        crypten.init()
        self.ttp_group = comm.get().ttp_group
        self.comm_group = comm.get().ttp_comm_group
        self.device = "cuda"
        self._setup_generators()

        logging.info("TTPServer Initialized")
        try:
            while True:
                # Wait for next request from client
                message = comm.get().recv_obj(0, self.ttp_group)
                logging.info("Message received: %s" % message)

                if message == "terminate":
                    logging.info("TTPServer shutting down.")
                    return

                function = message["function"]
                args = message["args"]
                kwargs = message["kwargs"]
                result = getattr(self, function)(*args, **kwargs)

                if function == "additive":
                    comm.get().send_obj(result.size(), 1, self.ttp_group)
                comm.get().send_obj(result.size(), 0, self.ttp_group)
                comm.get().broadcast(result, 2, self.comm_group)
        except RuntimeError:
            print(RuntimeError)
            logging.info("Encounter RuntimeError. TTPServer shutting down.")

    def _setup_generators(self):
        """Create random generator to send to a party"""
        ws = comm.get().get_world_size()

        seeds = [torch.randint(-(2 ** 63), 2 ** 63 - 1, size=()) for _ in range(ws)]
        reqs = [
            dist.isend(tensor=seeds[i], dst=i, group=self.ttp_group) for i in range(ws)
        ]
        self.generators = [torch.Generator(device=self.device) for _ in range(ws)]

        for i in range(ws):
            self.generators[i].manual_seed(seeds[i].item())
            reqs[i].wait()

        dist.barrier(group=self.ttp_group)

    def _get_additive_PRSS(self, size, remove_rank=False):
        """
        Generates a plaintext value from a set of random additive secret shares
        generated by each party
        """
        gens = self.generators[1:] if remove_rank else self.generators
        result = torch_stack(
            [
                generate_random_ring_element(size, generator=g, device=g.device)
                for g in gens
            ]
        )
        return result.sum(0)

    def _get_binary_PRSS(self, size, bitlength=None, remove_rank=None):
        """
        Generates a plaintext value from a set of random binary secret shares
        generated by each party
        """
        gens = self.generators[1:] if remove_rank else self.generators
        result = [
            generate_kbit_random_tensor(
                size, bitlength=bitlength, generator=g, device=g.device
            )
            for g in gens
        ]
        return reduce(lambda a, b: a ^ b, result)

    def additive(self, size0, size1, op, *args, **kwargs):

        # Add all shares of `a` and `b` to get plaintext `a` and `b`
        a = self._get_additive_PRSS(size0)
        b = self._get_additive_PRSS(size1)

        c = getattr(torch, op)(a, b, *args, **kwargs)

        # Subtract all other shares of `c` from plaintext value of `c` to get `c0`
        c0 = c - self._get_additive_PRSS(c.size(), remove_rank=True)
        return c0

    def square(self, size):
        # Add all shares of `r` to get plaintext `r`
        r = self._get_additive_PRSS(size)
        r2 = r.mul(r)
        return r2 - self._get_additive_PRSS(size, remove_rank=True)

    def binary(self, size0, size1):
        # xor all shares of `a` and `b` to get plaintext `a` and `b`
        a = self._get_binary_PRSS(size0)
        b = self._get_binary_PRSS(size1)

        c = a & b

        # xor all other shares of `c` from plaintext value of `c` to get `c0`
        c0 = c ^ self._get_binary_PRSS(c.size(), remove_rank=True)
        return c0

    def wraps(self, size):
        r = [generate_random_ring_element(size, generator=g) for g in self.generators]
        theta_r = count_wraps(r)

        return theta_r - self._get_additive_PRSS(size, remove_rank=True)

    def B2A(self, size):
        rB = self._get_binary_PRSS(size, bitlength=1)

        # Subtract all other shares of `rA` from plaintext value of `rA`
        rA = rB - self._get_additive_PRSS(size, remove_rank=True)

        return rA

    def rand(self, *sizes, encoder=None):
        if encoder is None:
            encoder = FixedPointEncoder()  # use default precision

        r = encoder.encode(torch.rand(*sizes, device=self.device))
        r = r - self._get_additive_PRSS(sizes, remove_rank=True)
        return r
