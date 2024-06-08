# M-E-GA
By running this code, you acknowledge and agree to the terms of the [License Agreement](LICENSE.txt). 


# Introduction

With corporate influence over Artificial Intelligence and Machine Learning growing more and more every day, I believe it is essential for there to be ML projects brought into the sphere of public control and ownership. The Mutable Encoding Enabled Genetic Algorithm (MEGA) is intended as a foundational first step towards the development of advanced Artificial Intelligence as a public asset.

MEGA is a passion project of mine that I have been working on for a very long time. It represents years of thought, consideration, and study; 20 in fact (more than half my life).

So what's so special about it?

Well, MEGA is a GA but not a GA. It includes several elements that I have always found lacking in a traditional GA. Mainly, for an algorithm that is supposed to model evolution to solve problems, traditional GA doesn't do a very good job of employing biology-inspired problem solving. There is much more involved in evolution than just the transfer of genetic information from one organism to another. Biology constructs and preserves structured genetic information in the form of proteins, which themselves consist of smaller structures called exons, and then the next step below that is the individual base pairs of amino acids. Traditional Genetic Algorithms have no way of constructing or preserving the patterns they discover while exploring the search space. In fact, most of the information about the search space is thrown away from one generation to the next.

MEGA introduces several new ideas that work together to create a fundamentally different approach to doing Genetic Algorithms. As opposed to a complex adaptive system, I view MEGA as a singular adaptive system where the parts come together and work to define and enable the whole.

The new concepts that are introduced:

- **Start and End Delimiters:** They work to demarcate regions of a solution that are eligible to be captured to create new meta-genes.
- **Meta Genes:** A single gene that is a composite of delimited genes after a capture. They can be further nested within other captures to create a nested hierarchy of meta-genes. They should be thought of as a sub-solution.
- **Capture Mutation:** Capture is a novel mutation that serves as the trigger for the creation of a new meta-gene.
- **Open Mutation:** Open is a novel mutation that triggers the decompression of a meta-gene back to its pre-capture state. If a meta-gene is opened in an undelimited region, then the delimiters are replaced around it. Otherwise, if a meta-gene is opened inside a delimited region, then there are no delimiters placed because this will disrupt the ordering of delimiters causing issues. Open gives meta-genes the ability to be refined and improved on before being recaptured for use.
- **Insert/Delete Delimiter Pair Mutations:** These either insert a pair of delimiters or delete a pair of delimiters. Insert does not allow the pairs that it inserts to intrude on another delimiter pair and will either not place the delimiters or place them so as not to disrupt an existing pair.
- **Point/Insert Mutations:** Point and insert mutations work in the typical way, however when choosing a gene to use, there is a set probability to favor inserting either a meta-gene or a base gene. This is to prevent meta-genes from overwhelming the base genes and thus preventing further exploration of the search space. There is also a kind of aging parameter called Capture_gene_prob (yes, I need a better name for that). It is used so that as the number of meta-genes increases, older genes are less favored for selection, making it more likely that a newer meta-gene will be selected for use. Introducing this helped reduce over-exploring due to the uniform chance of meta-gene selection. This caused a very large fitness distribution within the population, though it didn’t seem to impact the ability to find a new highest fitness individual at regular intervals. Try setting it to 0 and see what happens.
- **Base Genes:** The initial genes that the solutions decode back into. They are, as of right now, discrete strings only. Yes, this needs further development, but this is what they are at the moment.

These new elements come together to create what I refer to as a Meta Evolution of the gene representation. The meta-genes themselves are subject to selective pressure, in that they can be opened and refined. Meta-genes that do not contribute to fitness are pruned with the organisms that contain them, thus don’t have the opportunity to propagate or be opened and create new meta-genes. This is essentially a second evolutionary process applied to the gene representation. Since the representation is available to be incorporated in the population as a whole, the population not only exists to evolve the solution but to evolve a functional collective genome. This is why I view this as a singular adaptive system (SAS) as opposed to a complex adaptive system (CAS).

The nesting of meta-genes creates a structure that I now know to be called a Directed Acyclic Graph (DAG). This is the hierarchy of meta-genes and is essentially what the algorithm learns about the search space through its lifecycle. This meta-gene structure can be saved and used to initialize another run of the GA, facilitating knowledge transfer. MEGA instances that receive knowledge from a previous run show faster increases in fitness and overall higher fitness levels, showing that the knowledge gained in the previous run is adaptable and can be improved on.

From here, everything operates the same as a traditional Genetic Algorithm.

I apologize for updates being a little erratic. I am balancing this between a full-time non-academic job and a family. This is my first open-source project and I am learning every day. I hope that I will have time on my "weekend" this Monday and Tuesday to get things more organized and in line with what is expected out of a professional repository. Thank you for your patience. I hope you have the chance to play around with this more. Documentation and properly commented code will be coming very soon.

Thanks,
Matt

I can be reached at avilanch2000@yahoo.com or at the MEGA Discord server [https://discord.gg/jQWRCwrj](https://discord.gg/jQWRCwrj)
