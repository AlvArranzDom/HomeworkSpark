import main.PreprocessCSV.preprocessCSV
import main.StudyTrainModels.studyTrainModel
import org.apache.spark.sql.SparkSession
import java.nio.file.{Files, Paths}
import org.clapper.argot._
import ArgotConverters._

object SparkAssignment {

  private val parser = new ArgotParser("Cato", preUsage = Some("Spark Assignment, Version 0.1. Copyright (c) 2019, UPM."))

  // this line supports `-f inputPath` and `--file inputPath`
  private val inputFileParser = parser.option[String](List("f", "file"), "file", "Input File Path .")
  // this line supports `-m modelName` and `--model modelName`
  private val modelNameParser = parser.option[String](List("m", "model"), "model", "Model to be trained.")


  def main(args: Array[String]): Unit = {

    try {
      var inputPath = ""
      var modelName = ""
      parser.parse(args)
      inputFileParser.value match {
        case Some(file) => inputPath = file
      }
      modelNameParser.value match {
        case Some(model) => modelName = model
      }

      val spark = SparkSession.builder
        .master("local[*]")
        .appName("HomeworkSpark")
        .getOrCreate

      val sc = spark.sparkContext
      val sqlc = spark.sqlContext
      spark.sparkContext.setLogLevel("ERROR")

      if (Files.exists(Paths.get(inputPath))) {
        if (modelName == "lr" || modelName == "gbtr" || modelName == "rfr" || modelName == "rfc") {
          val startTimeMillis = System.currentTimeMillis()

          println("Preprocessing & Normalizing Dataset...")
          val normalizedCSVPath = preprocessCSV(sqlc, inputPath)

          println("Study Variables & Training the model...")
          studyTrainModel(sqlc, normalizedCSVPath, modelName)

          val endTimeMillis = System.currentTimeMillis()
          val durationSeconds = (endTimeMillis - startTimeMillis) / 1000
          val residual_s = durationSeconds % 60
          val residual_m = (durationSeconds / 60) % 60

          println(s"Study & Train Model finished in ${residual_m} minutes and ${residual_s} seconds.")
        } else {
          println(s"Model don't implemented to be trained.\nPlease review the doc to know which models are available.")
        }
      } else {
        println(s"Input CSV does not exists.")
      }

      sqlc.clearCache()
      sc.clearCallSite()
      sc.clearJobGroup()
      spark.close()
    } catch {
      case e: ArgotUsageException => println(e.message)
    }
  }
}
