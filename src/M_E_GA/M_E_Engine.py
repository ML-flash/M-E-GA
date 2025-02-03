import random
import xxhash
import functools
from collections import OrderedDict

class EncodingManager:
    def __init__(self, lru_cache_size=100, logger=None):
        self.logger = logger  # New: accept a logger for research/real-time metrics
        self.encodings = {}
        self.reverse_encodings = {}
        self.meta_genes = []         # Existing references rely on this list
        self.meta_gene_stack = []    # NEW: track newest->oldest metagenes
        self.gene_counter = 3
        self.lru_cache_size = lru_cache_size
        self.metagene_usage = OrderedDict()
        self.deletion_basket = {}  # {metagene_id: generations_unused}
        self.unused_encodings = []
        self.current_generation = 0
        self.debug = False

        # Add default delimiters
        self.add_gene('Start', predefined_id=1)
        self.add_gene('End', predefined_id=2)

    def get_metagene_status(self):
        samples = []
        if self.debug and self.meta_genes:
            sample_size = min(3, len(self.meta_genes))
            if sample_size > 0:
                for hash_key in random.sample(self.meta_genes, sample_size):
                    if hash_key in self.encodings:
                        samples.append({
                            'id': hash_key,
                            'content': self.decode(self.encodings[hash_key]),
                            'in_lru': hash_key in self.metagene_usage,
                            'in_basket': hash_key in self.deletion_basket,
                        })

        return {
            'generation': self.current_generation,
            'total_metagenes': len(self.meta_genes),
            'in_basket': len(self.deletion_basket),
            'unused': len(self.unused_encodings),
            'in_lru': len(self.metagene_usage),
            'samples': samples
        }

    def generate_hash_key(self, identifier):
        return xxhash.xxh64_intdigest(str(identifier))

    def add_gene(self, gene, verbose=False, predefined_id=None):
        if gene in self.reverse_encodings:
            return self.reverse_encodings[gene]

        if self.unused_encodings and predefined_id is None:
            hash_key = self.unused_encodings.pop(0)
        else:
            identifier = predefined_id if predefined_id is not None else self.gene_counter
            hash_key = self.generate_hash_key(identifier)
            if predefined_id is None:
                self.gene_counter += 1

        self.encodings[hash_key] = gene
        self.reverse_encodings[gene] = hash_key
        return hash_key

    def update_metagene_usage(self, hash_key):
        if hash_key not in self.meta_genes:
            return

        # Remove from deletion basket if used
        self.deletion_basket.pop(hash_key, None)

        # Update LRU cache
        if hash_key in self.metagene_usage:
            self.metagene_usage.move_to_end(hash_key)
        else:
            if len(self.metagene_usage) >= self.lru_cache_size:
                # Move least recently used to deletion basket
                lru_key, _ = self.metagene_usage.popitem(last=False)
                if lru_key not in self.deletion_basket:
                    self.deletion_basket[lru_key] = 0
            self.metagene_usage[hash_key] = True

    def start_new_generation(self):
        self.current_generation += 1

        # First pass: Identify metagenes to delete
        to_delete = []
        for hash_key, gen_count in list(self.deletion_basket.items()):
            if gen_count >= 2:
                to_delete.append(hash_key)
                if self.debug:
                    print(f"Marking metagene {hash_key} for deletion (unused for {gen_count} generations)")
            else:
                self.deletion_basket[hash_key] = gen_count + 1
                if self.debug:
                    print(f"Incrementing counter for metagene {hash_key} to {gen_count + 1}")

        # Second pass: Delete marked metagenes
        for hash_key in to_delete:
            self.delete_metagene(hash_key)

    # NEW helper methods to keep meta_genes and meta_gene_stack in sync
    def add_meta_gene(self, hash_key):
        """
        Adds the metagene to both self.meta_genes and self.meta_gene_stack.
        The newest metagene will appear at the end of meta_gene_stack.
        """
        if hash_key not in self.meta_genes:
            self.meta_genes.append(hash_key)
        if hash_key not in self.meta_gene_stack:
            self.meta_gene_stack.append(hash_key)  # newest at the end

    def remove_meta_gene(self, hash_key):
        """
        Removes the metagene from both self.meta_genes and self.meta_gene_stack
        so that ordering for meta_gene_stack remains correct.
        """
        if hash_key in self.meta_genes:
            self.meta_genes.remove(hash_key)
        if hash_key in self.meta_gene_stack:
            self.meta_gene_stack.remove(hash_key)

    def delete_metagene(self, hash_key):
        if hash_key not in self.meta_genes:
            return

        # Get top-layer contents
        target_contents = list(self.encodings[hash_key])

        # Update all metagenes that reference this one
        for meta_key in list(self.meta_genes):
            if meta_key != hash_key and isinstance(self.encodings[meta_key], tuple):
                meta_contents = list(self.encodings[meta_key])
                modified = False

                i = 0
                while i < len(meta_contents):
                    if meta_contents[i] == hash_key:
                        meta_contents[i:i + 1] = target_contents
                        modified = True
                        i += len(target_contents)
                    else:
                        i += 1

                if modified:
                    self.encodings[meta_key] = tuple(meta_contents)

        # Remove references
        # Instead of self.meta_genes.remove(hash_key), call remove_meta_gene
        self.remove_meta_gene(hash_key)

        self.metagene_usage.pop(hash_key, None)
        self.deletion_basket.pop(hash_key, None)
        self.encodings.pop(hash_key, None)

        # Add to unused pool
        if hash_key not in self.unused_encodings:
            self.unused_encodings.append(hash_key)

        if self.logger:
            self.logger.log_event("metagene_deleted", {
                "hash_key": hash_key,
                "generation": self.current_generation,
                "unused_pool_size": len(self.unused_encodings)
            })

        if self.debug:
            print(f"Deleted metagene {hash_key}")

    def capture_metagene(self, encoded_segment, verbose=False):
        if not encoded_segment:
            return False

        segment_tuple = tuple(encoded_segment)

        # Check for existing identical metagene
        for meta_id in self.meta_genes:
            if self.encodings[meta_id] == segment_tuple:
                self.update_metagene_usage(meta_id)
                return meta_id

        # Get available ID
        if self.unused_encodings:
            hash_key = self.unused_encodings.pop(0)
            if self.logger:
                self.logger.log_event("metagene_reused", {
                    "hash_key": hash_key,
                    "generation": self.current_generation,
                    "unused_pool_size": len(self.unused_encodings)
                })
        else:
            hash_key = self.generate_hash_key(self.gene_counter)
            self.gene_counter += 1

        # Store new metagene
        self.encodings[hash_key] = segment_tuple

        # Instead of self.meta_genes.append(hash_key), call add_meta_gene
        self.add_meta_gene(hash_key)

        self.update_metagene_usage(hash_key)

        if verbose:
            print(f"Captured new metagene with ID {hash_key}")

        if self.logger:
            self.logger.log_event("metagene_captured", {
                "hash_key": hash_key,
                "segment": segment_tuple,
                "generation": self.current_generation
            })

        return hash_key

    def open_metagene(self, hash_key, no_delimit=False, verbose=False):
        if hash_key not in self.encodings or not isinstance(self.encodings[hash_key], tuple):
            return [hash_key]

        decompressed = []
        if not no_delimit:
            decompressed.append(self.reverse_encodings['Start'])

        decompressed.extend(self.encodings[hash_key])

        if not no_delimit:
            decompressed.append(self.reverse_encodings['End'])

        return decompressed

    def encode(self, genes, verbose=False):
        encoded_list = []
        for gene in genes:
            hash_key = self.reverse_encodings.get(gene)
            if hash_key is None:
                if verbose:
                    print(f"Gene '{gene}' is not recognized.")
                continue
            encoded_list.append(hash_key)
        return encoded_list

    @functools.lru_cache(maxsize=1000)
    def decode(self, encoded_tuple, verbose=False):
        if not encoded_tuple:
            return []

        if not isinstance(encoded_tuple, tuple):
            encoded_tuple = (encoded_tuple,)

        stack = list(encoded_tuple)
        decoded_sequence = []

        while stack:
            hash_key = stack.pop(0)
            if hash_key in self.encodings:
                value = self.encodings[hash_key]
                self.update_metagene_usage(hash_key)

                if isinstance(value, tuple):
                    stack = list(value) + stack
                else:
                    decoded_sequence.append(value)
            else:
                decoded_sequence.append("Unknown")

        return decoded_sequence

    def generate_random_organism(self, functional_length=100, include_specials=False,
                                 special_spacing=10, probability=0.99, verbose=False):
        gene_pool = [gene for gene in self.reverse_encodings if gene not in ['Start', 'End']]
        organism_genes = [random.choice(gene_pool) for _ in range(functional_length)]
        special_gene_indices = set()

        if include_specials:
            for i in range(len(organism_genes)):
                if random.random() <= probability:
                    if all(abs(i - idx) >= special_spacing for idx in special_gene_indices):
                        organism_genes.insert(i, 'Start')
                        end_index = min(i + special_spacing, len(organism_genes))
                        organism_genes.insert(end_index, 'End')
                        special_gene_indices.update([i, end_index])

        return self.encode(organism_genes, verbose=verbose)

    def integrate_uploaded_encodings(self, uploaded_encodings, base_genes, verbose=False):
        if isinstance(uploaded_encodings, str):
            uploaded_encodings = {int(k): v for k, v in (item.split(':') for item in uploaded_encodings.split(','))}

        start_key = self.reverse_encodings.get('Start')
        end_key = self.reverse_encodings.get('End')

        for key, value in uploaded_encodings.items():
            if value in base_genes or key in [start_key, end_key]:
                if value not in self.reverse_encodings or key in [start_key, end_key]:
                    self.encodings[key] = value
                    self.reverse_encodings[value] = key
            elif isinstance(value, tuple):
                self.encodings[key] = value
                # Because this is a metagene, add it to both meta_genes and meta_gene_stack
                self.add_meta_gene(key)
                self.metagene_usage[key] = True

        max_hash_key = max(self.encodings.keys(), default=0)
        self.gene_counter = max(self.gene_counter, max_hash_key + 1)
