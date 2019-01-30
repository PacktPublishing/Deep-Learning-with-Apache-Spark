package chapter4;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.jetbrains.annotations.NotNull;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;


public class MultiGpuClassificationExample {

  public static void main(String[] args) throws Exception {
    configureCudaEnvironemnt();

    int nChannels = 1;
    int outputNumOfClasses = 10;

    int batchSize = 128; //for GPU more batches
    int nEpochs = 10;
    int seed = 123;

    System.out.println("Load data....");
    DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);
    DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, 12345);

    System.out.println("Build model....");
    MultiLayerNetwork model = createMultiLayerNetwork(nChannels, outputNumOfClasses, seed);
    ParallelWrapper wrapper = configureGPUsage(model);

    performClassification(nEpochs, mnistTrain, model, wrapper);

    evaluateModel(outputNumOfClasses, mnistTest, model);
  }

  private static void performClassification(int nEpochs, DataSetIterator mnistTrain, MultiLayerNetwork model, ParallelWrapper wrapper) {
    System.out.println("Train model....");
    model.setListeners(new ScoreIterationListener(100));
    long timeX = System.currentTimeMillis();

    for (int i = 0; i < nEpochs; i++) {
      long time1 = System.currentTimeMillis();
      wrapper.fit(mnistTrain); //fit on ParallelWrapper not the model.fit. ParallelWrapper will call or model underneath
      long time2 = System.currentTimeMillis();
      System.out.println("*** Completed epoch: " + i + ", time: " + (time2 - time1));
    }
    long timeY = System.currentTimeMillis();

    System.out.println("*** Training complete, time: " + (timeY - timeX));
  }

  private static void evaluateModel(int outputNumOfClasses, DataSetIterator mnistTest, MultiLayerNetwork model) {
    System.out.println("Evaluate model....");
    Evaluation eval = new Evaluation(outputNumOfClasses);
    while (mnistTest.hasNext()) {
      DataSet ds = mnistTest.next();
      INDArray output = model.output(ds.getFeatures(), false);
      eval.eval(ds.getLabels(), output);
    }
    System.out.println(eval.stats());
    mnistTest.reset();
  }

  private static ParallelWrapper configureGPUsage(MultiLayerNetwork model) {
    return new ParallelWrapper.Builder(model)// ParallelWrapper perform load balancing between GPUs.
        .prefetchBuffer(24) // Set it to be equal to number of GPUs on which training is done
        .workers(2) // set to number of physical devices (or x2)
        .averagingFrequency(3)//rare averaging improves performance, but reduce model accuracy
        .reportScoreAfterAveraging(true)//On every iteration of the model the score will be printed
        .build();
  }

  @NotNull
  private static MultiLayerNetwork createMultiLayerNetwork(int nChannels, int outputNumOfClasses, int seed) {
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(seed)
        .l2(0.0005)
        .weightInit(WeightInit.XAVIER)
        .updater(new Nesterovs.Builder().learningRate(.01).build())
        .biasUpdater(new Nesterovs.Builder().learningRate(0.02).build())
        .list()
        .layer(0, new ConvolutionLayer.Builder(5, 5)
            //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
            .nIn(nChannels)
            .stride(1, 1)
            .nOut(20)
            .activation(Activation.IDENTITY)
            .build())
        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
            .kernelSize(2, 2)
            .stride(2, 2)
            .build())
        .layer(2, new ConvolutionLayer.Builder(5, 5)
            //Note that nIn need not be specified in later layers
            .stride(1, 1)
            .nOut(50)
            .activation(Activation.IDENTITY)
            .build())
        .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
            .kernelSize(2, 2)
            .stride(2, 2)
            .build())
        .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
            .nOut(500).build())
        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nOut(outputNumOfClasses)
            .activation(Activation.SOFTMAX)
            .build())
        .setInputType(InputType.convolutionalFlat(28, 28, 1)) //See note below
        .backprop(true).pretrain(false).build();
    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();
    return model;
  }

  private static void configureCudaEnvironemnt() {
    CudaEnvironment.getInstance().getConfiguration()
        // key option enabled
        .allowMultiGPU(true)

        // we're allowing larger memory caches
        .setMaximumDeviceCache(2L * 1024L * 1024L * 1024L)

        // cross-device access is used for faster model averaging over pcie
        .allowCrossDeviceAccess(true);
  }
}
