## Example Data

**Warning:** This data is specific to MIMIC II. However, to apply to other datasets, transform your data to match the files described below:

- ICD9_descriptions -- A tab separated file with the ICD9 code and its description
- all_labels_final.txt -- All the MIMIC II ICD9 IDs.
- lookup_ids_labels.txt -- Pipe delimited file matching MIMIC II IDs to ICD9 codes.
- example_train.json -- A JSON file with 3 keys
    - labels -- A list of MIMIC II ICD9 IDs
    - txt -- The MIMIC II discharge text
    - id -- The example id
- example_dev.json -- The dev set format (same as example_train.json)
- mimic2_adj_matrix.pkl -- A xz compressed numpy matrix. This matrix is an adjacency matrix for all codes in all_labels_final.txt. Please note that the order of the rows/columns of the adjacency matrix must match the order in all_labels_final.txt.
