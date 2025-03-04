# ROADMAP

1. **Typed Genes**
    - Currently, out genome only represents labels, not discrete values, which is a problem. We need to brainstorm various possible ways to adapt our algorithm to represent arbitrary typed data into the genome.
    - Using ints and floats as a gene name isn't an option (can't stringify typed data, or we'd lose the range that numbers are capable of). We decided to completely steer away from that approach.
    - We need to think how other GA libraries are doing it, and see if we can adapt that to our current algorithm.
    - One of the goal of having typed genes is so we can evolve a Neural Network, storing variables such as connection weights into the genome to leverage the evolution algorithm.
    - Add multidimensional typed genes (like vectors for neural network weights).
    - Implement numeric mutations (e.g., Gaussian, Cauchy).
