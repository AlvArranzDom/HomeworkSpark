# HomeworkSpark

## How to execute

1. Execute PreprocessCSVScala. This process will follow the next steps:
    1. Clean Initial Information that is mandatory to delete.
    2. Cast columns to prepare the dataset.
    3. Filter to get the non cancelled flights and the non NA & Null values.
    4. Makes to indexing process:
        1. Index string categorizing them.
        2. Index integer between 1 and 0, taking the maximum and minimum.
    5. Reorder the information putting in the last place the response variable.
    6. Save Preprocessed & Normalized Dataset into a CSV output directory.
2. Using the info generated from this Process, you will be able to analyze correlations.

## Next Steps

- Work with the information related from the correlation, trying to study more the dataset.
- Study the different models available in Spark Scala.
- Start training the models with the normalized dataset. (Exists the possibility of generates two normalized datasets of differents years and join them to have more training information.)