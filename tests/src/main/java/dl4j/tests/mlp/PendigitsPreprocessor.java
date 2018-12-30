package dl4j.tests.mlp;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.analysis.columns.ColumnAnalysis;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.CategoricalColumnCondition;
import org.datavec.api.transform.condition.column.DoubleColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.normalize.Normalize;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.misc.WritablesToStringFunction;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerType;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

import org.datavec.spark.transform.AnalyzeSpark;
import org.datavec.spark.transform.Normalization;

public class PendigitsPreprocessor {
	
	
	public static JavaRDD<List<Writable>> getFile(String name, JavaSparkContext sc) throws IOException {
        //Define the path to the data file. You could use a directory here if the data is in multiple files
        //Normally just define your path like "file:/..." or "hdfs:/..."
        String pathTrain= new ClassPathResource(name).getFile().getAbsolutePath();
        JavaRDD<String> stringDataTrain = sc.textFile(pathTrain);
   
        
        //We first need to parse this format. It's comma-delimited (CSV) format, so let's parse it using CSVRecordReader:
        RecordReader rr = new CSVRecordReader();
        JavaRDD<List<Writable>> parsedInputDataTrain = stringDataTrain.map(new StringToWritablesFunction(rr));
        
        return parsedInputDataTrain;
		
	}
	
    public static  void main(String[] args) throws Exception {

     
        Schema inputDataSchema = new Schema.Builder()
        		.addColumnInteger("X1")      		
        		.addColumnInteger("Y1")
        		.addColumnInteger("X2")      		
        		.addColumnInteger("Y2")
        		.addColumnInteger("X3")      		
        		.addColumnInteger("Y3")
        		.addColumnInteger("X4")      		
        		.addColumnInteger("Y4")
        		.addColumnInteger("X5")      		
        		.addColumnInteger("Y5")
        		.addColumnInteger("X6")      		
        		.addColumnInteger("Y6")
        		.addColumnInteger("X7")      		
        		.addColumnInteger("Y7")
        		.addColumnInteger("X8")      		
        		.addColumnInteger("Y8")
        		.addColumnInteger("CLASS")
        		.build();
        
        
        SparkConf conf = new SparkConf();
        conf.setMaster("local[*]");
        conf.setAppName("DataVec Example");

        JavaSparkContext sc = new JavaSparkContext(conf);

//        //Define the path to the data file. You could use a directory here if the data is in multiple files
//        //Normally just define your path like "file:/..." or "hdfs:/..."
//        String pathTrain= new ClassPathResource("pendigits.tra").getFile().getAbsolutePath();
//        JavaRDD<String> stringDataTrain = sc.textFile(pathTrain);
//
//        
//        String pathTest= new ClassPathResource("pendigits.tes").getFile().getAbsolutePath();
//        JavaRDD<String> stringDataTest = sc.textFile(pathTest);
   
        
        //We first need to parse this format. It's comma-delimited (CSV) format, so let's parse it using CSVRecordReader:
//        RecordReader rr = new CSVRecordReader();
        
        System.out.println("TRA");
        JavaRDD<List<Writable>> parsedInputDataTrain = getFile("pendigits.tra", sc);
        System.out.println("TES");
        JavaRDD<List<Writable>> parsedInputDataTest = getFile("pendigits.tes", sc);

        
        DataAnalysis analysis = AnalyzeSpark.analyze(inputDataSchema, parsedInputDataTrain, 12);
        System.out.println(analysis);
        
        
        
        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                //Let's remove some column we don't need
                .normalize("X1", Normalize.MinMax2, analysis)
                .normalize("Y1", Normalize.MinMax2, analysis)
                .normalize("X2", Normalize.MinMax2, analysis)
                .normalize("Y2", Normalize.MinMax2, analysis)
                .normalize("X3", Normalize.MinMax2, analysis)
                .normalize("Y3", Normalize.MinMax2, analysis)
                .normalize("X4", Normalize.MinMax2, analysis)
                .normalize("Y4", Normalize.MinMax2, analysis)
                .normalize("X5", Normalize.MinMax2, analysis)
                .normalize("Y5", Normalize.MinMax2, analysis)
                .normalize("X6", Normalize.MinMax2, analysis)
                .normalize("Y6", Normalize.MinMax2, analysis)
                .normalize("X7", Normalize.MinMax2, analysis)
                .normalize("Y7", Normalize.MinMax2, analysis)
                .normalize("X8", Normalize.MinMax2, analysis)
                .normalize("Y8", Normalize.MinMax2, analysis)
                .build();
        
        
        
        
        //Now, let's execute the transforms we defined earlier:
        JavaRDD<List<Writable>> processedDataTrain = SparkTransformExecutor.execute(parsedInputDataTrain, tp);
        JavaRDD<String> processedAsStringTrain = processedDataTrain.map(new WritablesToStringFunction(","));
        List<String> processedCollectedTrain = processedAsStringTrain.collect(); 
        
        JavaRDD<List<Writable>> processedDataTest = SparkTransformExecutor.execute(parsedInputDataTest, tp);
        JavaRDD<String> processedAsStringTest = processedDataTest.map(new WritablesToStringFunction(","));
        List<String> processedCollectedTest = processedAsStringTest.collect(); 
 
        
        //For the sake of this example, let's collect the data locally and print it:
        //processedAsString.saveAsTextFile("file://your/local/save/path/here");   //To save locally
        //processedAsString.saveAsTextFile("hdfs://your/hdfs/save/path/here");   //To save to hdfs

        
        //Normalization.minMaxColumns(rr, "X2");
        //List<String> inputDataCollected = stringDataTrain.collect();


        System.out.println("\n\n---- Original Data ----");
        //for(String s : inputDataCollected) System.out.println(s);

        System.out.println("\n\n---- Processed Data ----");
        //for(String s : processedCollected) System.out.println(s);


        System.out.println("\n\nDONE");
    }
        		
        

}
