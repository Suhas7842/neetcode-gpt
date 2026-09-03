from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        # 1. Split corpus into individual characters
        tokens = list(corpus)

        merges = []

        # 2. Perform merge operations
        for _ in range(num_merges):

            # Count adjacent pairs
            pair_counts = {}

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])

                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            # No pairs left to merge
            if not pair_counts:
                break

            # Most frequent pair
            # Lexicographically smallest pair breaks ties
            best_pair = min(
                pair_counts,
                key=lambda pair: (-pair_counts[pair], pair)
            )

            # Record the merge
            merges.append([best_pair[0], best_pair[1]])

            # Merge non-overlapping occurrences
            new_tokens = []
            i = 0

            while i < len(tokens):

                if (
                    i < len(tokens) - 1
                    and (tokens[i], tokens[i + 1]) == best_pair
                ):
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return merges
