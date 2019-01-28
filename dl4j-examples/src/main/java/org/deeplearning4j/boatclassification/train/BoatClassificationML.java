package org.deeplearning4j.boatclassification.train;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BoatClassificationML {

	private static final Logger log = LoggerFactory.getLogger(BoatClassificationML.class);

	private static final String dataDirTrain = "C:\\Users\\yuceef\\Downloads\\Classification\\train\\";
	private static final String csvTrain = "C:\\Users\\yuceef\\Downloads\\Classification\\train_ship_segmentations.csv";
	private static final String dataDirTest = "C:\\Users\\yuceef\\Downloads\\Classification\\test\\";
	private static final String csvTest = "C:\\Users\\yuceef\\Downloads\\Classification\\test_ship_segmentations.csv";

	public static void main(String[] args) throws Exception {
		int height = 28;
		int width = 28;
		int channels = 3; // single channel for grayscale images
		int outputNum = 2; 
		int batchSize = 128;
		int nEpochs = 1;

		int seed = 123456;
		Random randNumGen = new Random(seed);
		
		log.info("Preparing Data ...");
		PreparedData preparedDataTrain = new PreparedData(csvTrain, dataDirTrain);
		preparedDataTrain.organizedData();
		PreparedData preparedDataTest = new PreparedData(csvTest, dataDirTest);
		preparedDataTest.organizedData();
		
		// vectorization of train data
		log.info("vectorization of train data ...");
		File trainData = new File(dataDirTrain);
		FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // parent path as the image label
		ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
		trainRR.initialize(trainSplit);
		DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);

		// pixel values from 0-255 to 0-1 (min-max scaling)
		DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
		scaler.fit(trainIter);
		trainIter.setPreProcessor(scaler);

		// vectorization of test data
		log.info("vectorization of test data ...");
		File testData = new File(dataDirTest);
		FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
		ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
		testRR.initialize(testSplit);
		DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
		testIter.setPreProcessor(scaler); // same normalization for better results

		log.info("Network configuration and training...");
		Map<Integer, Double> lrSchedule = new HashMap<>();
		lrSchedule.put(0, 0.06); // iteration #, learning rate
		lrSchedule.put(200, 0.05);
		lrSchedule.put(600, 0.028);
		lrSchedule.put(800, 0.0060);
		lrSchedule.put(1000, 0.001);

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).l2(0.0005)
				.updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION, lrSchedule)))
				.weightInit(WeightInit.XAVIER).list()
				.layer(0,
						new ConvolutionLayer.Builder(5, 5).nIn(channels).stride(1, 1).nOut(20)
								.activation(Activation.IDENTITY).build())
				.layer(1,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
								.kernelSize(2, 2).stride(2, 2).build())
				.layer(2, new ConvolutionLayer.Builder(5, 5).stride(1, 1) // nIn need not specified in later layers
						.nOut(50).activation(Activation.IDENTITY).build())
				.layer(3,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build())
				.layer(4, new DenseLayer.Builder().activation(Activation.RELU).nOut(500).build())
				.layer(5,
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(outputNum)
								.activation(Activation.SOFTMAX).build())
				.setInputType(InputType.convolutionalFlat(height, width, channels)) // InputType.convolutional for
																					// normal image
				.backprop(true).pretrain(false).build();
		log.info("UIServer Init....");
		UIServer uiServer = UIServer.getInstance();
		StatsStorage statsStorage = new InMemoryStatsStorage();
		uiServer.attach(statsStorage);
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));
		log.debug("Total num of params: {}", net.numParams());

		// evaluation while training (the score should go down)
		for (int i = 0; i < nEpochs; i++) {
			net.fit(trainIter);
			log.info("Completed epoch {}", i);
			Evaluation eval = net.evaluate(testIter);
			log.info(eval.stats());
			trainIter.reset();
			testIter.reset();
		}

		ModelSerializer.writeModel(net, new File("boat-model.zip"), true);
	}
}
