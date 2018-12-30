package dl4j.tests.mlp;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.evaluation.classification.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.File;

/**
 * "Saturn" Data Classification Example, adapted from
 *  https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/classification/MLPClassifierSaturn.java
 *
 * Based on the data from 
 * https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/
 * 
 * @author Josh Patterson
 * @author Alex Black (added plots)
 *
 */
public class Pendigits {


    public static void main(String[] args) throws Exception {
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        int batchSize = 50;
        int seed = 123;
        double learningRate = 0.01;
        //Number of epochs (full passes of the data)
        int nEpochs = 1000;

        int numInputs = 2;
        int numOutputs = 2;
        int numHiddenNodes = 33;

        final String filenameTrain  = new ClassPathResource("pendigits.tra").getFile().getPath();
        final String filenameTest  = new ClassPathResource("pendigits.tes").getFile().getPath();
        
        DataNormalization normalizer = new NormalizerMinMaxScaler(-1, 1);

        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(filenameTrain)));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,16,10);

        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(filenameTest)));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,16,10);
        

        normalizer.fit(trainIter);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        trainIter.setPreProcessor(normalizer);
        testIter.setPreProcessor(normalizer);
        
//        0.99
//        Train: 0.9890579129970644
//        Test : 0.9579759862778731
        
        float rate = 0.98f;
        
        
        //log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(16).nOut(numHiddenNodes).dropOut(rate)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes).dropOut(rate)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes).dropOut(rate)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes).nOut(10).build())
                .build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1));    //Print score every 10 parameter updates

        for ( int n = 0; n < nEpochs; n++) {
            model.fit( trainIter );
            //System.out.println("Score="+model.score());
            
            
            System.out.println("Epoch #" + n);
            if(n % 10 == 0) {
            	System.out.println("Evaluate model test....");
            	Evaluation eval = model.evaluate(testIter);
            	//System.out.println(eval.stats());
            	testIter.reset();
            	
            	//System.out.println("*************************** TRAIN ***************************");
                trainIter.reset();
                Evaluation eval1 = model.evaluate(trainIter);
                //System.out.println(eval1.stats());
                System.out.println("Train: "+eval1.accuracy());
                System.out.println("Test : "+eval.accuracy());

            }
        }

        System.out.println("Evaluate model....");
        Evaluation eval = model.evaluate(testIter);
 
        
        
        
//        Evaluation eval = new Evaluation(numOutputs);
//        while(testIter.hasNext()){
//            DataSet t = testIter.next();
//            INDArray features = t.getFeatures();
//            INDArray lables = t.getLabels();
//            INDArray predicted = model.output(features,false);
//
//            eval.eval(lables, predicted);
//
//        }


        System.out.println(eval.stats());

    }

}