package main.helpers

import org.apache.spark.ml.feature.{MinMaxScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.DoubleType

/**
 * Object that has all the functions related to the transformation and preparation of the
 * dataset for training ML models.
 */
object DataFrameFunctions {

  /**
   * Function that normalized a column with the StringIndexer method in a specific dataset.
   *
   * @param df       entry dataset to be analyzed and modified.
   * @param inputCol column to apply StringIndexer method
   * @return modified dataset
   */
  private def stringIndexer(df: DataFrame, inputCol: String): DataFrame = {
    val indexer = new StringIndexer()
      .setInputCol(inputCol)
      .setOutputCol(inputCol + "Scaled")

    val indexed_DF = indexer.fit(df)
      .transform(df)
      .drop(inputCol)
      .withColumnRenamed(inputCol + "Scaled", inputCol)

    indexed_DF
  }

  /**
   * Function that normalized a column with the MinMaxScaler method in a specific dataset.
   *
   * @param df       entry dataset to be analyzed and modified.
   * @param inputCol column to apply MinMaxScaler method
   * @return modified dataset
   */
  private def minMaxScaler(df: DataFrame, inputCol: String): DataFrame = {
    val vectorizeCol = udf((v: Double) => Vectors.dense(Array(v)))
    val headValue = udf((arr: DenseVector) => arr.toArray(0))
    val newDF = df.withColumn(inputCol + "Vec", vectorizeCol(df(inputCol)))

    val scalerArrDelay = new MinMaxScaler()
      .setInputCol(inputCol + "Vec")
      .setOutputCol(inputCol + "Scaled")
      .setMax(1)
      .setMin(0)

    val scaled_DF = scalerArrDelay.fit(newDF)
      .transform(newDF)
      .drop(inputCol)
      .drop(inputCol + "Vec")
      .withColumnRenamed(inputCol + "Scaled", inputCol)

    scaled_DF.withColumn(inputCol, headValue(col(inputCol)).cast(DoubleType))
  }

  /**
   * Function that normalized the entry dataset spliting the variables between the StringType ones above the others.
   *
   * @param df entry dataset to be normalized and modified.
   * @return modified dataset
   */
  private def normalizeStringColumnsDataFrame(df: DataFrame): DataFrame = {
    var normalized_DF = df
    val colNamesList = normalized_DF.columns
    val columnDataTypes: Array[String] = normalized_DF.schema.fields.map(x => x.dataType).map(x => x.toString)

    for (i <- colNamesList.indices) {
      val colName = colNamesList(i)
      val colDataType = columnDataTypes(i)

      if (colName != "label") {
        if (colDataType == "StringType") {
          normalized_DF = stringIndexer(normalized_DF, colName)
        }
      }
    }

    normalized_DF
  }

  /**
   * Function that normalized the entry dataset spliting the variables between the StringType ones above the others.
   *
   * @param df entry dataset to be normalized and modified.
   * @return modified dataset
   */
  private def normalizeNumericColumnsDataFrame(df: DataFrame): DataFrame = {
    var normalized_DF = df
    val colNamesList = normalized_DF.columns
    val columnDataTypes: Array[String] = normalized_DF.schema.fields.map(x => x.dataType).map(x => x.toString)

    for (i <- colNamesList.indices) {
      val colName = colNamesList(i)
      val colDataType = columnDataTypes(i)

      if (colName != "label") {
        if (colDataType != "StringType") {
          normalized_DF = minMaxScaler(normalized_DF, colName)
        }
      }
    }

    normalized_DF
  }

  /**
   * Function that calculate and evaluate the correlations. If the correlations meets the threshold we save his name,
   * otherwise we drop the column from the dataset. Finally, we generate the Features Vector with the function
   * VectorAssembler.
   *
   * @param df        entry dataset to be analyzed and modified.
   * @param threshold acceptance positive value for the correlations
   * @return modified dataset
   */
  private def calculateAndEvaluateCorrelations(df: DataFrame, threshold: Double): (DataFrame, Array[String]) = {
    var filtered_DF = df
    val colNamesList = df.columns
    var featuresColsNames = Array[String]()

    for (i <- colNamesList.indices) {
      val colName = colNamesList(i)

      if (colName != "label") {
        var correlation = df.stat.corr("label", colName)
        correlation = Math.abs(correlation)

        if (correlation < threshold) {
          filtered_DF = filtered_DF.drop(colName)
        } else {
          println(s"correlation between Prediction Variable and $colName = $correlation")
          featuresColsNames = colName +: featuresColsNames
        }
      }
    }

    (filtered_DF, featuresColsNames)
  }

  /**
   * Function that prepares a dataset to be used in Models.
   *
   * @param df           entry dataset to be analyzed and modified.
   * @param labelColName label column of the dataset.
   * @param threshold    acceptance value for the correlations
   * @return modified dataset
   */
  def createDataFrameForModel(df: DataFrame, labelColName: String, threshold: Double = 0.3): DataFrame = {
    var filtered_DF = df.withColumnRenamed(labelColName, "label")

    // Normalize DataFrame String Columns using StringIndexer
    filtered_DF = normalizeStringColumnsDataFrame(filtered_DF)

    // Drop the columns with low correlation value
    val (clean_DF, featuresColsNames) = calculateAndEvaluateCorrelations(filtered_DF, threshold)

    // Normalize DataFrame Numeric Columns using MinMaxScales
    val normalized_DF = normalizeNumericColumnsDataFrame(clean_DF)

    // Create Features Vector
    val assembler = new VectorAssembler().setInputCols(featuresColsNames).setOutputCol("features")
    val model_DF = assembler.transform(normalized_DF)

    model_DF
  }
}
