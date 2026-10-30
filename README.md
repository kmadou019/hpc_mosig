# GPU edit distance between DNA sequences (CUDA)

Course project (HPC, MOSIG master, Grenoble, February 2025): computing the **edit distance between two genetic sequences** with the Needleman-Wunsch algorithm, and parallelizing it on GPU with **CUDA**.

The starting point is a sequential, recursive implementation with memoization provided by the course (Jean-Louis Roch, Ensimag). This repository contains my GPU version and its measurement tools.

## Approach

Needleman-Wunsch fills an (M+1) × (N+1) dynamic-programming table, where each cell depends on its left, top and top-left neighbours. A naive row-by-row loop is sequential, so the GPU version reformulates the computation of a row:

1. **`pre_compute_min`** (CUDA kernel): for each cell of the row, computes in parallel the candidate cost coming from the previous row (substitution or deletion).
2. **Parallel prefix scan** (`prescan`, work-efficient exclusive scan in shared memory): propagates the insertion costs along the row, which turns the left-to-right dependency into a scan operation that runs in O(log N) steps.
3. The row is then complete, and the next one starts from it.

## Repository structure

| Path | Content |
|------|---------|
| `src/distanceEdition.cu` | Main program: reads two FASTA sequences and prints their edit distance |
| `src/Needleman-Wunsch-recmemo.cu` | Needleman-Wunsch implementation and the `pre_compute_min` kernel |
| `src/utils.cu` | Parallel prefix scan (`prescan`) kernel |
| `srcperf/` | Time, CPU and energy measurement (Intel RAPL counters) |
| `tests/` | Test sequences (including SARS-CoV-2 Wuhan-Hu-1 and Omicron BA.5.2 genomes) and expected outputs |

## Build and run

Requirements: an NVIDIA GPU, the CUDA toolkit (`nvcc`) and `gcc`.

```bash
make
./bin/distanceEdition ./tests/f1.fna 0 5 ./tests/f2.fna 42 7
# expected output: 7
```

Arguments: `file1 start1 length1 file2 start2 length2`, i.e. the sub-sequence of each FASTA file to compare.

The `Makefile` also builds a version instrumented with the measurement tools of `srcperf/`, and runs the tests in `tests/`.
