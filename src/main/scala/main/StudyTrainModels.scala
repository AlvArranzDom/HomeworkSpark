package main

import main.helpers.DataFrameFunctions.createDataFrameForModel
import main.helpers.TrainingModelsFunctions.trainModel
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}

object StudyTrainModels {
  val THRESHOLD = 0.20

  def studyTrainModel(sqlc: SQLContext, normalized_DF: DataFrame, modelName: String): Unit = {
    var df = normalized_DF

    if (modelName == "lr" || modelName == "gbtr" || modelName == "rfr") {
      df = df.drop("ArrDelayStatus")

      val lr_model_DF = createDataFrameForModel(df, "ArrDelay", THRESHOLD)
      trainModel(modelName, lr_model_DF)

    } else if (modelName == "rfc") {
      df = df.drop("ArrDelay")

      val classification_model_DF = createDataFrameForModel(df, "ArrDelayStatus", THRESHOLD)
      trainModel(modelName, classification_model_DF)

    } else {
      println(s"Model don't implemented to be trained. \n Please review the doc to know which models are available.")
    }
  }
}
