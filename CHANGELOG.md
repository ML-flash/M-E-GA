# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Changed
- **Refactored Terminology in `M_E_GA_Base` and `ExperimentRunner`**:
  - **Renamed Probability Parameters in `M_E_GA_Base`**:
    - `'open_mutation_prob'` → `'open_metagene_mutation_prob'`
    - `'capture_mutation_prob'` → `'capture_metagene_mutation_prob'`
    - `'delimiter_insert_prob'` → `'insert_delimiter_pair_prob'`
    - `'delimit_delete_prob'` → `'delete_delimiter_prob'`
    - `'capture_gene_prob'` → `'capture_metagene_prob'`
  - **Updated `ga_config` in `ExperimentRunner`**:
    - Updated the `ga_config` dictionary to reflect the renamed probability parameters:
      - `'open_mutation_prob'` → `'open_metagene_mutation_prob'`
      - `'capture_mutation_prob'` → `'capture_metagene_mutation_prob'`
      - `'delimiter_insert_prob'` → `'insert_delimiter_pair_prob'`
      - `'delimit_delete_prob'` → `'delete_delimiter_prob'`
      - `'capture_gene_prob'` → `'capture_metagene_prob'`
  - **Renamed Methods and Variables in `M_E_GA_Base`**:
    - **Methods**:
      - `capture_segment` → `capture_metagene`
      - `open_segment` → `open_metagene`
      - `mutate_organism` → `mutate_metagene`
    - **Variables**:
      - `captured_segments` → `meta_genes`
      - Introduced `meta_genome` to represent the collection of all meta genes.
  - **Updated Logging Statements and Documentation**:
    - Ensured all logging statements within both `M_E_GA_Base` and `ExperimentRunner` use the new terminology (`meta_genes`, `meta_genome`, etc.).
    - Updated inline comments and docstrings to reflect the renaming and refactoring.

- **Updated Mutation Logic in `M_E_GA_Base`**:
  - Adjusted mutation-related methods to utilize the newly renamed probability parameters.
  - Ensured that mutation operations (`capture_metagene`, `open_metagene`, etc.) correctly reference the updated parameters.
  - Improved clarity and consistency in mutation logging to reflect the new mutation types.

- **Enhanced Logging Structure**:
  - Added `meta_genome` to the final log in `M_E_GA_Base` to capture the collection of all meta genes.
  - Ensured that all mutation logs now reference `'capture_metagene'`, `'open_metagene'`, etc., instead of the old terms.

- **Compatibility Adjustments**:
  - Ensured that `ExperimentRunner` interacts seamlessly with the refactored `M_E_GA_Base` by updating parameter names and method calls.
  - Verified that the `NetworkEvolutionFitness` class is compatible with the new GA terminology, ensuring smooth integration and functionality.

- **Documentation and Comments**:
  - Updated inline comments and docstrings in both `M_E_GA_Base` and `ExperimentRunner` to reflect the new terminology and parameter names.
  - Enhanced clarity in code documentation to aid future maintenance and onboarding.

### Added
- **Comprehensive Refactoring Documentation**:
  - Detailed descriptions of the changes made to align with the new terminology.
  - Clear separation of concerns between `M_E_GA_Base` and `ExperimentRunner` regarding their roles and interactions with `meta_genes`.

### Fixed
- **Parameter Alignment**:
  - Corrected inconsistencies in parameter naming between `ExperimentRunner` and `M_E_GA_Base` to prevent runtime errors and ensure cohesive functionality.

### Notes
- These changes are part of ongoing refactoring efforts to improve code clarity, maintainability, and alignment with the conceptual framework of meta genes and the meta genome.
- No functionality changes were introduced beyond renaming and refactoring for consistency. All existing behaviors remain intact.
- Future releases will continue to build upon this refactored foundation, introducing new features and optimizations as needed.

## [1.0.0b3] - 2024-10-15
### Changed
- **Fixed a broken relative import** that snuck in and prevented the `M_E_GA_Base` from importing the `M_E_Engine`. Repackaging and re-uploading everything.

## [1.0.0b2] - 2024-10-15
### Changed
- **Fixed a broken relative import** that snuck in and prevented the `M_E_GA_Base` from importing the `M_E_Engine`. Repackaging and re-uploading everything.

### Notes
- No changes were made to the `M_E_Engine` in this release; all updates were focused on `M_E_GA_Base`.

## [1.0.0b1] - 2024-10-14
### Changed
- **Fixed a logging bug**: Resolved an issue in `M_E_GA_Base` where generations were logged twice—once empty and once with data—causing confusion and redundancy in the logs.
- **Refactored mutation logic**: Improved the mutation process in `M_E_GA_Base` by restructuring how mutation probabilities are applied. Previously, special mutations were applied first, which caused normal mutations to be underrepresented. Now, all mutations are weighted, ensuring only one probability roll per gene, with mutation types selected based on their assigned probabilities.
- **Added `delimit_delete_prob` parameter**: Introduced a new probability parameter `delimit_delete_prob` in `M_E_GA_Base` to control the likelihood of deleting delimiters during mutations.

### Notes
- No changes were made to the `M_E_Engine` in this release; all updates were focused on `M_E_GA_Base`.

## [1.0.0b0] - 2024-07-21
### Added
- **Core functionalities**: Initial implementation of the genetic algorithm framework.
- **`EncodingManager` class** in `M_E_Engine`:
  - Handles gene encoding and decoding.
  - Manages gene additions and reverse encodings.
  - Supports capturing and decompressing genetic segments.
  - Provides methods for generating random organisms.
- **`M_E_GA_Base` class**:
  - Implements the base structure for the genetic algorithm.
  - Features population initialization, fitness evaluation, selection, crossover, and mutation operations.
  - Supports detailed logging mechanisms for generations, mutations, crossovers, and individual organism states.
  - Includes customizable parameters for mutation probabilities, crossover rates, elitism ratio, and more.
- **Logging and Experiment Management**:
  - Comprehensive logging system to track the evolution of generations and mutations.
  - Ability to save logs to specified directories with timestamped filenames.
  - Supports custom experiment names for better organization.

### Notes
- This is the first beta release aimed at gathering user feedback and identifying potential improvements.
- The focus is on establishing a flexible genetic algorithm framework that can be extended and customized for various applications.

---
