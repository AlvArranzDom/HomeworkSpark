name := "HomeworkSpark"

version := "0.1"

scalaVersion := "2.11.12"

mainClass := Some("SparkAssignment")

// https://mvnrepository.com/artifact/org.apache.spark/spark-core
libraryDependencies += "org.clapper" %% "argot" % "1.0.4"
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.4"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.4"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.4.4"