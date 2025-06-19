# IFBLUP Data Format Specification

This document details the input data format requirements for the IFBLUP program.

## 1. Pedigree and Phenotype Data (CSV format)

Pedigree and phenotype data should be saved in CSV format, with each row representing an individual. The columns have the following meanings:

| Column # | Content | Example Values | Notes |
|--------|----------|--------|------|
| 1 | Individual ID | 1, 2, 3... | Must be unique integers |
| 2 | Father ID | 0, 5, 12... | 0 indicates unknown |
| 3 | Mother ID | 0, 34, 14... | 0 indicates unknown |
| 4 | Sex code | 0, 1 | Typically 0=female, 1=male |
| 5 | Generation | 0, 1, 2... | Base population usually 0 |
| 6-N | Fixed effects | 46, 54, 0... | Can have multiple fixed effect columns |
| N+1 | Phenotype value | 0.505419... | Continuous trait value |

### Example Data (First Few Lines)

```
1,0,0,0,0,46,54,0,0.505419,-0.013908,0.01269,0,-0.026598
2,0,0,0,0,47,53,0,0.50044,0.240286,0.89712,0,-0.656834
3,0,0,0,0,55,45,0,0.4981,-0.813248,0.209996,0,-1.023243
```

### Important Notes

- CSV files should not contain header rows
- All IDs must be integers
- Missing values should be represented as 0 (for ID fields)
- Phenotype values should be floating-point numbers

## 2. Genotype Data (PLINK Binary Format)

Genotype data should use PLINK's standard binary format, consisting of the following three files:

### .bed File

A binary file containing genotype data, with each SNP's genotype encoded as:
- 0: Homozygote (AA)
- 1: Heterozygote (AB)
- 2: Homozygote (BB)
- Missing value

### .bim File

A text file containing SNP information, with each line having 6 columns:
1. Chromosome number
2. SNP ID
3. Genetic distance (cM)
4. Physical position (bp)
5. Allele 1
6. Allele 2

### .fam File

A text file containing individual information, with each line having 6 columns:
1. Family ID
2. Individual ID (must match the ID in the pedigree CSV file)
3. Father ID
4. Mother ID
5. Sex (1=male, 2=female, 0=unknown)
6. Phenotype value (not used in this program, can be set to -9)

## 3. Data Preparation Recommendations

1. **Data Cleaning**: Ensure there are no duplicate IDs and pedigree relationships are logically consistent
2. **Genotype Quality Control**: Use PLINK for routine quality control, such as removing SNPs with low MAF or high missing rates
3. **ID Matching**: Ensure that genotyped individuals' IDs can be found in the pedigree data
4. **Missing Value Handling**: Individuals with missing phenotypes can be kept in the pedigree but will not participate in model training

## 4. Example Data

This project provides small-scale example data for testing:
- `5055quan.csv`: Contains pedigree and phenotype data for 5,055 individuals
- `5k.bed/bim/fam`: Contains genotype data for 5,000 SNPs

You can reference the format of these example datasets to prepare your own data.

## 5. Large Dataset Processing Recommendations

For large datasets (e.g., over 100,000 individuals or 500,000 SNPs), we recommend:

1. Ensure sufficient disk space for intermediate files
2. Consider using SSD storage to improve I/O speed
3. Provide adequate memory (at least 16GB recommended, possibly 64GB+ for very large datasets)
4. For extremely large datasets, consider running small-scale tests first to estimate resource requirements