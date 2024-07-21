# Mutable Encoding enabled Genetic Algorithm (MEGA)

## Description
With corporate influence over Artificial Intelligence and Machine Learning growing more and more every day, I believe it is essential for there to be ML projects brought into the sphere of public control and ownership. The Mutable Encoding Enabled Genetic Algorithm (MEGA) is intended as a foundational first step towards the development of advanced Artificial Intelligence as a public asset.

MEGA is a passion project of mine that I have been working on for a very long time. It represents years of thought, consideration, and study; 20 in fact (more than half my life). The aim of MEGA is to bring new ideas into the sphere of Evolutionary Algorithms. Through leveraging these new ideas, I hope to provide advancements in the field such as:

- Trivialized transfer learning.
- More biologically inspired computing models.
- Rapid iteration and deployment through a general-purpose encoding scheme and easily deployable framework.
- A lower barrier to entry in the form of domain-specific encodings.
- Providing new approaches to Evolutionary Algorithms in general, offering fresh paths for research and advancement. (hopefully)

## Index
1. [Description](#description)
2. [Key Points](#key-points)
3. [How to Install and Run MEGA](#how-to-install-and-run-mega)
4. [How to Use the MEGA](#how-to-use-mega)
5. -[Building Fitness Functions](#building-fitness-functions)
5. [Credits](#credits)
6. [License](#license)
7. [Badges](#badges)
8. [How to Contribute to the Project](#how-to-contribute-to-the-project)
9. [Tests](#tests)
10. [Buy Me a Coffee](https://ko-fi.com/mlflash)

## Key Points
- MEGA takes an entirely different approach to GA, making it both the same and fundamentally different from traditional approaches. It enables a Meta-Evolution of the gene representation. Through the continual capturing, nesting, and refinement of new meta-genes, MEGA enables the GA to learn about the search space in a very real tangible way that can be transferred to a newly initialized GA instance. This provides faster fitness gains that typically exceed the GA they originated from.
- xxhash is used to generate the hashable encodings. They aren't used in a hash table. I don't know enough about that to get it going. functools is used in the M_E_engine to allow for meta-gene caching using least recently used to enable faster decoding, mitigating the overhead created by nested meta-genes. concurrent is used to facilitate threading when evaluating a population. This helps again with organism decoding and fitness evaluation speed.
- The main challenges I encountered while developing MEGA were delimiter management. Start and End delimiters have a direction and not respecting that during mutations and crossover causes problems that are sometimes hard to detect. This was a pain, and if you look at the code, most of the helper functions are geared around ensuring delimiters are detected and handled properly. So if you start playing with things and everything seems to work but something is off, pass the verbose=True flag when decoding, and it will print out the values as it's decoding, and you can see if there are mismatched Start, End delimiters.

## How to Install and Run MEGA

### Prerequisites
Before you can install the `M_E_GA` package, make sure to have Python and `pip` installed on your system. If you are using a Conda environment, ensure it is activated.



### Installing M_E_GA

`M_E_GA` can be installed using `pip`, which will also handle installing the necessary dependencies. It is recommended to install this package within a virtual environment to avoid conflicts with other projects or system-wide packages.


#### Step 1: Ensure pip is up to date

pip install `--upgrade pip`

#### Step 2: Install M_E_GA using pip

`pip install M_E_GA` This command installs the latest version of M_E_GA along with its dependencies.

### Verifying Installation
`python -c "import M_E_GA; print('M_E_GA installed successfully!')"`

If there are no errors and you see the success message, M_E_GA is ready for use.


## How to Use MEGA

MEGA is designed to be modular and fairly easy to build on. The M_E_Engine is distinct from the M_E_GA_base, which is the Genetic Algorithm interface with the M_E_Engine via the EncodingManager class. Then, there are the individual experiments. These serve as a way to standardize how MEGA is operated. The GA is initialized and run through a user-defined set of processes, giving the ability to handle the population in nuanced ways, i.e., threading or adjusting fitness values. You could also build a simulation around an experiment together with the fitness function and create customized breeding scenarios. Say you are running an ALife simulation, and you have certain conditions where the population will breed. This can be defined in the fitness function and pass pairs back into the GA to breed before again using them within the fitness function simulation. In short, there is a lot of flexibility built into MEGA that allows for very nuanced, user-defined use case scenarios
### Building Fitness Functions
The fitness function is a key component of any Genetic Algorithm it is what defines what the GA is doing. MEGA has the fitness function abstracted from the Main GA allowing for the use of custom fitness functions that are easily created using following the below walk through. 

   ```sh
class LeadingOnesFitness:
   def __init__(self, max_length, update_best_func):
        self.max_length = max_length
        self.update_best = update_best_func  # Store the passed function for updating the best organism
        self.genes = ['0', '1']  # Specific genes for this fitness function
  ```
We begin with the Class definition. You can use any name for the call you just need to change it in the experiment you run it with. As of right now Fitness functions follow a specific format. 
 Since MEGA uses mutable encoding where the organism length is variable `max_length` is passed in so that the maximum length of the organism can be tracked and properly handled without needing to be explicitly set or modified in the fitness function with every new experiment.

`update_best_func` is a function that is defined in the experiment and is passed in on initialisation. It allows a simple straight forward way to keep track of the new best organism as it appears through the run. This is especially helpful when running an experiment that uses transfer learning As this allows you to track the Maximum fitness individual across all learning phases of the algorithm.

`self.genes = ['0', '1']`  At the moment MEGA uses genes in the form of a list of strings. Genes can be custom defined here for use within the compute method. This is how organisms are scored in as a fitness value. A solution is a list of genes that represent the values being optimized for by the GA this will be discussed below in the compute method.

Alternatively if you need dynamically generated genes you could run a function at initialisation and return the string gene values.

```sh
    def compute(self, encoded_individual, ga_instance):
        # Decode the individual
        decoded_individual = ga_instance.decode_organism(encoded_individual)
```
`ga_instance.decode_organism(encoded_individual)` takes in the encoded organism and returns the decoded individual for fitness evaluation. There is a `format` flag. Pass it as `True` to remove delimiters from the decoded organism. 

From here you can use the `decoded_individual` to calculate your fitness sore and return it as an int or float.

```sh
        # Initialize fitness score
        fitness_score = 0

        # Count the number of leading '1's until the first '0'
        for gene in decoded_individual:
            if gene == '1':
                fitness_score += 1
            else:
                break  # Stop counting at the first '0'

        # Calculate the penalty
        if len(decoded_individual) < self.max_length:
            penalty = 1.008 ** (self.max_length - len(decoded_individual))
        else:
            penalty = len(decoded_individual) - self.max_length

        # Compute final fitness score
        final_fitness = fitness_score - penalty

        # Update the best organism using the passed function
        self.update_best(encoded_individual, final_fitness, verbose=True)

        # Return the final fitness score after applying the penalty
        return final_fitness
   ```
This is the essentials of building a MEGA compatable fitness function.

Coming soon we will be a link to a break down of designing a custom experiment and eventually a Break down of the M_E_GA_Base Class. 




## Credits



## License
By running this code, you acknowledge and agree to the terms of the [License Agreement](LICENSE.txt). 

## Badges


## How to Contribute to the Project



## Tests
After installing the M_E_GA package, you can run the included tests to verify that everything is working correctly on your system. Here's how you can run the tests:

1. Navigate to the directory where the tests are located (this might be inside the package directory if installed, or you can download the tests from the source repository if not included in the installation).

2. Use the following command to run the tests using Python's unittest framework:

I can be reached at matthew.andrews2024@gmail.com or at the MEGA [Discord](https://discord.gg/VXA3Jj3hcv) server.


[Buy Me a Coffee](https://ko-fi.com/mlflash)