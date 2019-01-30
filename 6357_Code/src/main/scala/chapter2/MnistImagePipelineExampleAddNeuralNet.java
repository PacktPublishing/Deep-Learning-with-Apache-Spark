package chapter2;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class MnistImagePipelineExampleAddNeuralNet {
  // image information
  private static final int height = 28;
  private static final int width = 28;
  private static final int channels = 1; /// grayscale - single channel

  // training parameters
  private static final int rngseed = 123;
  private static final Random randNumGen = new Random(rngseed);
  private static final int batchSize = 128;
  private static final int outputNum = 10; //number of output classes - clusters
  private static final int numEpochs = 1;

  /** Location to save and extract the training/testing data */
  public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_Mnist/");

  public static void main(String[] args) throws Exception {

    //download data from http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
    DataUtilities.downloadData();

    File trainData = new File(DATA_PATH + "/mnist_png/training");
    File testData = new File(DATA_PATH + "/mnist_png/testing");

    FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
    FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

    ParentPathLabelGenerator labelMaker = extractImageLabel();

    ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
    recordReader.initialize(train);

    // DataSet Iterator
    DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
    DataNormalization scaler = setImageScaler(dataIter);

    MultiLayerNetwork model = buildNeuralNetwork();

    model.setListeners(new ScoreIterationListener(10));

    System.out.println("TRAIN MODEL");
    for (int i = 0; i < numEpochs; i++) {
      model.fit(dataIter);
    }

    System.out.println("EVALUATE MODEL");
    recordReader.reset();

    DataSetIterator testIter = validateModel(test, recordReader, scaler);

    //log 10 labels
    System.out.println(recordReader.getLabels().toString());

    // Create Eval object with 10 possible classes (clusters)
    Evaluation eval = new Evaluation(outputNum);

    // Evaluate the network
    while (testIter.hasNext()) {
      DataSet next = testIter.next();
      INDArray output = model.output(next.getFeatures());
      // Compare the Feature Matrix from the model
      // with the labels from the RecordReader
      eval.eval(next.getLabels(), output);
    }

    System.out.println(eval.stats());
  }

  @NotNull
  private static DataSetIterator validateModel(FileSplit test, ImageRecordReader recordReader, DataNormalization scaler) throws IOException {
    // evaluate against the test data of images the network has not seen yet
    recordReader.initialize(test);
    DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
    scaler.fit(testIter);
    testIter.setPreProcessor(scaler);
    return testIter;
  }

  @NotNull
  private static MultiLayerNetwork buildNeuralNetwork() {
    System.out.println("BUILD MODEL");
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(rngseed)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .updater(new Nesterovs(0.006, 0.9))
        .l2(1e-4)
        .list()
        .layer(0, new DenseLayer.Builder()
            .nIn(height * width)
            .nOut(100)
            .activation(Activation.RELU)
            .weightInit(WeightInit.XAVIER)
            .build())
        .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nIn(100)
            .nOut(outputNum)
            .activation(Activation.SOFTMAX)
            .weightInit(WeightInit.XAVIER)
            .build())
        .setInputType(InputType.convolutional(height, width, channels))
        .build();

    return new MultiLayerNetwork(conf);
  }

  @NotNull
  private static DataNormalization setImageScaler(DataSetIterator dataIter) {
    // Scale pixel values to 0-1
    DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    scaler.fit(dataIter);
    dataIter.setPreProcessor(scaler);
    return scaler;
  }

  @NotNull
  private static ParentPathLabelGenerator extractImageLabel() {
    return new ParentPathLabelGenerator();
  }

}