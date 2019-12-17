# HomeworkSpark

In this project is used the Scala version `2.11.12`.

## Project Dependencies

 - Argot (`v1.0.4`)
 - Spark Core Library (`v2.4.4`)
 - Spark SQL Library (`v2.4.4`)
 - Spark MLlib Library (`v2.4.4`)
 
## Project Structure

```
│-- README.md
│-- .gitignore
│-- .gitattributes
│-- build.sbt    
│
└───project
│   │-- build.properties
│   
└───src
    │   
    └───main
        │   
        └───scala
            │-- SparkAssignment.scala -> Main Object to execute the application.
            │   
            └───main
                │-- PreprocessCSV.scala -> Main Object to preprocess and prepare dataframe.
                │-- StudyTrainModels.scala -> Main Object to study and train ML Models.
                │
                └───helpers
                    │-- DataFrameFunctions.scala -> Object with functions to modify dataframes.
                    │-- TrainingModelsFunctions.scala -> Object with functions to train ML models.
```

## How to Install SBT
#### Installing sbt on macOS
-  Homebrew
```
    $ brew install sbt
```
-  SDKMAN!
```
    $ sdk install sbt
```
#### Installing sbt on Windows
##### Install JDK 
- Follow the link to install [JDK 8 or 11](https://adoptopenjdk.net/).

#####  Installing from a universal package 
- Download [ZIP](https://piccolo.link/sbt-1.3.4.zip) or [TGZ](https://piccolo.link/sbt-1.3.4.tgz) package and expand it.

##### Windows installer 
- Download [msi installer](https://piccolo.link/sbt-1.3.4.msi) and install it.

#### Installing sbt on Linux

- To install both JDK and sbt, consider using [SDKMAN](https://sdkman.io/).
```
    $ sdk list java
    $ sdk install java 11.0.4.hs-adpt
    $ sdk install sbt
```

## How to execute the project

After installing SBT, execute the next commands:
```
    $ cd HomeworkSpark
    $ sbt compile
    $ sbt "run -f file_path -m model_name"
```

## How the project works

1. Execute **PreprocessCSV**. This scala object will follow the next steps:
    1. Clean Initial Information that is mandatory to delete.
    2. Cast columns to prepare the dataset.
    3. Filter to get the non cancelled flights and the non NA & Null values.
    4. Makes to indexing process:
        1. Index string categorizing them.
        2. Index integer between 1 and 0, taking the maximum and minimum.
    5. Reorder the information putting in the last place the response variable.
    6. Save Preprocessed & Normalized Dataset into a CSV output directory.
2. Execute **StudyTrainModels**. This scala object will use the previous Dataset, it will follow the next steps:
    1. Cast columns to prepare the dataset.
    2. Create the final dataframe by:
        1. Analyzing Correlations
        2. Drop non acceptable columns by its correlation value, comparing it with a threshold.
        3. Create the Features Vector by using the Vector Assembler.
    3. Train the selected model providing final dataframe and model name to be trained (See available Models in next section)

## Available Models To Train

#### Regression
 - Linear Regression (`modelName = 'lr'`)`
 - Random Forest Regression (`modelName = 'gbtr'`)`
 - Gradient Boosting Tree Regression (`modelName = 'rfr'`)`
 
#### Classification
 - Random Forest Classifier (`modelName = 'rfc'`)