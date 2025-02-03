# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

# [2.0.0-b0] - 2025-2-2

## Added

### Meta-gene deletion in `M_E_Engine`
- Meta-genes are now automatically deleted after remaining unused for a specified number of generations.
- This process is managed through the `deletion_basket` and Least Recently Used (LRU) cache logic.

### Updated `no_delimit` flag in `open_segment` (Breaking Change)
- The `open_segment` method was modified to change how the `no_delimit` flag functions.
- **Previous Behavior**: `no_delimit` would open meta-genes without delimiters.
- **New Behavior**: It places `End` at the start of the opened segment and `Start` at the end, ensuring the newly opened segment remains outside of delimiters.
- **Impact**: 
  - Opened nested meta-genes are now more exposed to modification.
  - This change creates new undelimited space and counteracts the compressive pressure of the capture mutation.
  - Prevents stagnation in the evolutionary process by enabling greater flexibility.

### Introduced `metagene_stack` in the `EncodingManager`
- Ensures proper meta-gene ordering.
- Since the deletion process allows for recycling of meta-genes, the `encodings` and `reverse_encodings` lists may become inaccurate.
- This update affects the `select_gene` mechanism in `M_E_GA_Base`, which selects meta-genes based on their age (favoring older or newer ones), making order critical.

---

## Changed

### Refactored Parameter Usage (Breaking Change)
- `metagene_prob` was previously referred to as `capture_gene_prob`.
- This refactor ensures clearer terminology in alignment with the meta-gene framework.
- **Potential Fix**: Update existing configurations and code to replace `capture_gene_prob` with `metagene_prob`.

### Population Initialization (Breaking Change)
- `initialize_population` was updated to allow better control over special gene inclusion and spacing.
- **Potential Fix**: Ensure any custom calls to `initialize_population` are updated to handle the new arguments or defaults.

---

## Notes
### ⚠️ Breaking Changes
- This version introduces changes that may break dependent code, especially in:
  - Parameter naming (`capture_gene_prob` → `metagene_prob`).
  - `EncodingManager` initialization.
  - Meta-gene handling.

### Focus of this Update:
- Enhanced mutation handling.
- Improved evolutionary adaptability in MEGA.

**⚠️ Ensure dependent code and configurations are updated accordingly to avoid issues.**

## [1.0.0b3] - 2024-10-15
### Changed
- **Fixed a broken relative import that snuck in and prevented the M_E_GA_Base from importing the M_E_Engine. Repackaging and re-uploading everything.


## [1.0.0b2] - 2024-10-15
### Changed
- **Fixed a broken relative import that snuck in and prevented the M_E_GA_Base from importing the M_E_Engine. Repackaging and re-uploading everything.

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
