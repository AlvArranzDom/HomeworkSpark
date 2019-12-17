import main.PreprocessCSV.preprocessCSV
import main.StudyTrainModels.studyTrainModel
import org.apache.spark.sql.SparkSession
import java.nio.file.{Files, Paths}

object SparkAssignment extends App{

  override def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder
      .master("local[*]")
      .appName("HomeworkSpark")
      .getOrCreate

    val sc = spark.sparkContext
    val sqlc = spark.sqlContext
    spark.sparkContext.setLogLevel("ERROR")

    val inputPath = args(0)
    if (Files.exists(Paths.get(inputPath))) {
      val modelName = args(1).toString

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
  }
}
