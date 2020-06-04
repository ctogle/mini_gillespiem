# mini_gillespiem
Scalable Simulation Framework for Chemical Networks

This repository implements a framework for scalable simulation of chemical networks via the [Gillespie algorithm](https://en.wikipedia.org/wiki/Gillespie_algorithm).

It offers a fast mechanism for generating a model-specific Cython function to simulate the system and uses that function 
to scan parameters and compute ensemble-based results, optionally scaling to multiple cores/machines via MPI.

It was used in [Proteolytic Queues at ClpXP Increase Antibiotic Tolerance](https://pubs.acs.org/doi/abs/10.1021/acssynbio.9b00358) to simulate a simple network.
