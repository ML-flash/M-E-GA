# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
-  Next release will include deleting Meta Genes from the M_E_Engine in a LRU ( Least Recently Used) order. Still a work in progress.

## [1.0.0b2] - 2024-10-15
### Changed
- **Fixed a broken relative import that snuck in and prevented the M_E_GA_Base from importing the M_E_Engine. Repackaging and re-uploading everything

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
