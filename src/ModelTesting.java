import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.style.Styler;
import org.knowm.xchart.style.Styler.LegendPosition;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class ModelTesting {

	public static enum MODE {
		FUNDAMENTAL, ESSENTIAL, HOMOGRAPHY, ROTATION
	}

	public interface LabelMaker {
		public int getLabel(FinalizedData fd);
	}

	public static void trainModelFromScratch(String trainingFilename, String testingFilename, MODE mode,
			boolean excludePureRotation) {

		long EPOCHS = 100000;

		SimpleDateFormat formatter = new SimpleDateFormat("dd-MM-yyyy_HH-mm-ss");
		Date date = new Date();

		String MODELS_PATH = "results/models/updated_models_sept_15/";
		String MODEL_FILE_NAME = "logreg_real_" + mode.name() + "_" + formatter.format(date);

		// load training data
		List<FinalizedData> pretrain = loadData(trainingFilename);
		List<FinalizedData> train = new ArrayList<FinalizedData>();
		for (int i = 0; i < pretrain.size(); i++) {
			if (!excludePureRotation || pretrain.get(i).baseline > 0) {
				train.add(pretrain.get(i));
			}
		}

		// generate labels
		LabelMaker labeler;
		if (mode == MODE.FUNDAMENTAL) {
			Utils.pl("training for fundamental matrix case.");
			labeler = new LabelMaker() {
				public int getLabel(FinalizedData fd) {
					return getLabelFundamental(fd);
				}
			};
		} else if (mode == MODE.ESSENTIAL) {
			Utils.pl("training for essential matrix case.");
			labeler = new LabelMaker() {
				public int getLabel(FinalizedData fd) {
					return getLabelEssential(fd);

				}
			};
		} else if (mode == MODE.HOMOGRAPHY) {
			Utils.pl("training for homography case.");
			labeler = new LabelMaker() {
				public int getLabel(FinalizedData fd) {
					return getLabelHomography(fd);

				}
			};
		} else if (mode == MODE.ROTATION) {
			Utils.pl("training for pure rotation case.");
			labeler = new LabelMaker() {
				public int getLabel(FinalizedData fd) {
					return getLabelRotation(fd);
				}
			};
		} else {
			// default to fundamental
			Utils.pl("defaulting to fundamental matrix case.");
			labeler = new LabelMaker() {
				public int getLabel(FinalizedData fd) {
					return getLabelFundamental(fd);
				}
			};
		}

		INDArray labels = Nd4j.zeros(train.size(), 1);
		for (int i = 0; i < train.size(); i++) {
			labels.putScalar(new int[] { i, 0 }, labeler.getLabel(train.get(i)));
		}

		// create input array (normalize data)
		INDArray input = Nd4j.zeros(train.size(), 23);
		for (int i = 0; i < train.size(); i++) {
			INDArray row = Nd4j.create(train.get(i).summary.getArray());
			input.putRow(i, row);
		}

		// configure model
		Utils.pl("configuring model...");
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new Sgd(1)).seed(0)
				.weightInit(WeightInit.XAVIER).list().layer(new OutputLayer.Builder(LossFunction.XENT)
						.activation(Activation.SIGMOID).nIn(23).nOut(1).build())
				.build();
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		model.setListeners(new ScoreIterationListener(100));

		// train model
		Utils.pl("training (" + EPOCHS + " EPOCHS)...");
		DataSet data = new DataSet(input, labels);
		for (long i = 0; i < EPOCHS; i++) {
			model.fit(data);
		}

		// load testing data
		List<FinalizedData> pretest = loadData(testingFilename);
		List<FinalizedData> test = new ArrayList<FinalizedData>();
		for (int i = 0; i < pretest.size(); i++) {
			if (!excludePureRotation || pretest.get(i).baseline > 0) {
				test.add(pretest.get(i));
			}
		}

		// generate test labels
		INDArray testLabels = Nd4j.zeros(test.size(), 1);
		for (int i = 0; i < test.size(); i++) {
			testLabels.putScalar(new int[] { i, 0 }, labeler.getLabel(test.get(i)));
		}

		// create input array (normalize data)
		INDArray testInput = Nd4j.zeros(test.size(), 23);
		for (int i = 0; i < test.size(); i++) {
			INDArray row = Nd4j.create(test.get(i).summary.getArray());
			testInput.putRow(i, row);
		}

		// set up test data
		DataSet testData = new DataSet(testInput, testLabels);

		// create output for every training sample
		INDArray output = model.output(testData.getFeatures());
		System.out.println(output);

		// let Evaluation prints stats how often the right output had the highest value
		Evaluation eval = new Evaluation();
		eval.eval(testData.getLabels(), output);
		System.out.println(eval.stats());

		// save model
		try {
			String modelSave = MODELS_PATH + MODEL_FILE_NAME + (excludePureRotation ? "_rotExcluded" : "") + "-P"
					+ String.format("%.4f", eval.precision()) + "-R" + String.format("%.4f", eval.recall()) + "-F"
					+ String.format("%.4f", eval.f1()) + ".model";
			Utils.pl("saving model as " + modelSave);
			model.save(new File(modelSave));
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	// return label indicating whether or not the correspondences would be good for
	// a fundamental matrix estimate (1 is good for fundamental matrix, 0 is not)
	public static int getLabelFundamental(FinalizedData fd) {
		return fd.medianReconstErrorEstFun < 1 && fd.transChordalEstFun < 0.8 ? 1 : 0;
//		return fd.funNumGood / fd.summary.numCorrespondences > 0.9 && fd.funNumParallax > 50 ? 1 : 0;
	}

	// return label indicating whether or not the correspondences would be good for
	// a homography estimate (1 is good for homography, 0 is not)
	public static int getLabelHomography(FinalizedData fd) {
		return fd.medianReconstErrorEstHomography < 1 && fd.transChordalEstHomography < 0.8 ? 1 : 0;
//		return fd.homNumGood / fd.summary.numCorrespondences > 0.9 && fd.homNumParallax > 50 ? 1 : 0;
	}

	// return label indicating whether or not the correspondences would be good for
	// an essential matrix estimate (1 is good for essential matrix, 0 is not)
	public static int getLabelEssential(FinalizedData fd) {
		return fd.medianReconstErrorEstEssential < 1 && fd.transChordalEstEssential < 0.8 ? 1 : 0;
//		return fd.essNumGood / fd.summary.numCorrespondences > 0.9 && fd.essNumParallax > 50 ? 1 : 0;
	}

	// return label indicating whether or not the correspondences come from a pure
	// rotation (1 is pure rotation, 0 is not)
	public static int getLabelRotation(FinalizedData fd) {
		return fd.baseline <= 0.05 ? 1 : 0;
	}

	public static void generateTrainingData() {

		long seed = System.currentTimeMillis();

//		long seed = 1615348975802L;

		String OUT_FILE = "results/data/test-rev1-" + System.currentTimeMillis() + "-" + seed + ".dat";
		String serializedData = "";
		Random rand = new Random(seed);

		// randomly generate scenarios //
		int rejects = 0;
		int accepts = 0;
		int failed = 0;

		// high-parallax
		int highParallaxIterations = 1000;
		double maxBaseline = 2.5;
		double rotRange = 0.25;
		double rotOffset = rotRange / 2;
		VirtualEnvironment hpMock = new VirtualEnvironment();
//		hpMock.generateSphericalScene(seed, 1000);
		hpMock.generateScene0(seed);

		for (int i = 0; i < highParallaxIterations; i++) {

			Utils.pl("iteration: " + i);

			// decide camera movement
			double baseline = rand.nextDouble() * maxBaseline;
			double x = rand.nextDouble() - 0.5;
			double y = rand.nextDouble() - 0.5;
			double z = rand.nextDouble() - 0.5;
			double mag = Math.sqrt(x * x + y * y + z * z);
			x = x / mag * baseline;
			y = y / mag * baseline;
			z = z / mag * baseline;

			double rotX = rand.nextDouble() * rotRange - rotOffset;
			double rotY = rand.nextDouble() * rotRange - rotOffset;
			double rotZ = rand.nextDouble() * rotRange - rotOffset;

			hpMock.getSecondaryCamera().setCx(x);
			hpMock.getSecondaryCamera().setCy(y);
			hpMock.getSecondaryCamera().setCz(z);
			hpMock.getSecondaryCamera().setQw(1);
			hpMock.getSecondaryCamera().setQx(0);
			hpMock.getSecondaryCamera().setQy(0);
			hpMock.getSecondaryCamera().setQz(0);
			hpMock.getSecondaryCamera().rotateEuler(rotX, rotY, rotZ);

			Utils.pl("baseline: " + baseline);
			Utils.pl("x, y, z:");
			Utils.pl(x);
			Utils.pl(y);
			Utils.pl(z);
			Utils.pl("rotation:");
			Utils.pl(rotX);
			Utils.pl(rotY);
			Utils.pl(rotZ);

			Sample sample = new Sample();
			sample.evaluate(hpMock);
			Utils.pl("numCorrespondences: " + sample.correspondences.size());

			if (sample.correspondences.size() >= 10) {
				FinalizedData fd = new FinalizedData();
				fd.summary = sample.correspondenceSummary;
				fd.totalReconstErrorEstFun = sample.totalReconstErrorEstFun;
				fd.totalReconstErrorEstHomography = sample.totalReconstErrorEstHomography;
				fd.totalReconstErrorEstEssential = sample.totalReconstErrorEstEssential;
				fd.medianReconstErrorEstFun = sample.medianReconstErrorEstFun;
				fd.medianReconstErrorEstHomography = sample.medianReconstErrorEstHomography;
				fd.medianReconstErrorEstEssential = sample.medianReconstErrorEstEssential;
				fd.transChordalEstFun = sample.transChordalEstFun;
				fd.transChordalEstHomography = sample.transChordalEstHomography;
				fd.transChordalEstEssential = sample.transChordalEstEssential;
				fd.baseline = baseline;

				serializedData += fd.stringify();

				if (fd.totalReconstErrorEstFun / fd.summary.numCorrespondences > 1 || fd.transChordalEstFun > 0.4) {
					rejects++;
				} else {
					accepts++;
				}

			} else {
				failed++;
				Utils.pl("FAILED");
			}

			Utils.pl("");
		}

		// planar (noisy)
		Utils.pl("");
		Utils.pl("=====================================================================================");
		Utils.pl("====================================  PLANAR SCENE  =================================");
		Utils.pl("=====================================================================================");
		Utils.pl("");
		int planarIterations = 1000;
		maxBaseline = 2.5;
		rotRange = 0.25;
		rotOffset = rotRange / 2;
		VirtualEnvironment parallelMock = new VirtualEnvironment();
		parallelMock.generatePlanarScene(seed, 1000);

		for (int i = 0; i < planarIterations; i++) {

			Utils.pl("iteration: " + i);

			// decide camera movement
			double baseline = rand.nextDouble() * maxBaseline;
			double x = rand.nextDouble() - 0.5;
			double y = rand.nextDouble() - 0.5;
			double z = rand.nextDouble() - 0.5;
			double mag = Math.sqrt(x * x + y * y + z * z);
			x = x / mag * baseline;
			y = y / mag * baseline;
			z = z / mag * baseline;

			double rotX = rand.nextDouble() * rotRange - rotOffset;
			double rotY = rand.nextDouble() * rotRange - rotOffset;
			double rotZ = rand.nextDouble() * rotRange - rotOffset;

			parallelMock.getSecondaryCamera().setCx(x);
			parallelMock.getSecondaryCamera().setCy(y);
			parallelMock.getSecondaryCamera().setCz(z);
			parallelMock.getSecondaryCamera().setQw(1);
			parallelMock.getSecondaryCamera().setQx(0);
			parallelMock.getSecondaryCamera().setQy(0);
			parallelMock.getSecondaryCamera().setQz(0);
			parallelMock.getSecondaryCamera().rotateEuler(rotX, rotY, rotZ);

			Utils.pl("baseline: " + baseline);
			Utils.pl("x, y, z:");
			Utils.pl(x);
			Utils.pl(y);
			Utils.pl(z);
			Utils.pl("rotation:");
			Utils.pl(rotX);
			Utils.pl(rotY);
			Utils.pl(rotZ);

			Sample sample = new Sample();
			sample.evaluate(parallelMock);
			Utils.pl("numCorrespondences: " + sample.correspondences.size());

			if (sample.correspondences.size() >= 10) {
				FinalizedData fd = new FinalizedData();
				fd.summary = sample.correspondenceSummary;
				fd.totalReconstErrorEstFun = sample.totalReconstErrorEstFun;
				fd.totalReconstErrorEstHomography = sample.totalReconstErrorEstHomography;
				fd.totalReconstErrorEstEssential = sample.totalReconstErrorEstEssential;
				fd.medianReconstErrorEstFun = sample.medianReconstErrorEstFun;
				fd.medianReconstErrorEstHomography = sample.medianReconstErrorEstHomography;
				fd.medianReconstErrorEstEssential = sample.medianReconstErrorEstEssential;
				fd.transChordalEstFun = sample.transChordalEstFun;
				fd.transChordalEstHomography = sample.transChordalEstHomography;
				fd.transChordalEstEssential = sample.transChordalEstEssential;
				fd.baseline = baseline;

				serializedData += fd.stringify();

				if (fd.totalReconstErrorEstHomography / fd.summary.numCorrespondences > 1
						|| fd.transChordalEstHomography > 0.4) {
					rejects++;
				} else {
					accepts++;
				}

			} else {
				failed++;
				Utils.pl("FAILED");
			}

			Utils.pl("");
		}

		// pure rotation, high parallax
		double rotationAccepts = 0;
		double rotationRejects = 0;
		double rotationFails = 0;
		Utils.pl("");
		Utils.pl("=====================================================================================");
		Utils.pl("==========================  HIGH PARALLAX ROTATION SCENE  ===========================");
		Utils.pl("=====================================================================================");
		Utils.pl("");
		int hpRotationIterations = 1000;
		rotRange = 0.25;
		rotOffset = rotRange / 2;
		VirtualEnvironment hpRotationMock = new VirtualEnvironment();
		hpRotationMock.generateScene0(seed);

		for (int i = 0; i < hpRotationIterations; i++) {

			Utils.pl("iteration: " + i);

			// decide camera movement

			double rotX = rand.nextDouble() * rotRange - rotOffset;
			double rotY = rand.nextDouble() * rotRange - rotOffset;
			double rotZ = rand.nextDouble() * rotRange - rotOffset;

			hpRotationMock.getSecondaryCamera().setCx(0);
			hpRotationMock.getSecondaryCamera().setCy(0);
			hpRotationMock.getSecondaryCamera().setCz(0);
			hpRotationMock.getSecondaryCamera().setQw(1);
			hpRotationMock.getSecondaryCamera().setQx(0);
			hpRotationMock.getSecondaryCamera().setQy(0);
			hpRotationMock.getSecondaryCamera().setQz(0);
			hpRotationMock.getSecondaryCamera().rotateEuler(rotX, rotY, rotZ);

			Utils.pl("baseline: 0");
			Utils.pl("rotation:");
			Utils.pl(rotX);
			Utils.pl(rotY);
			Utils.pl(rotZ);

			Sample sample = new Sample();
			sample.evaluate(hpRotationMock);
			Utils.pl("numCorrespondences: " + sample.correspondences.size());

			if (sample.correspondences.size() >= 10) {
				FinalizedData fd = new FinalizedData();
				fd.summary = sample.correspondenceSummary;
				fd.totalReconstErrorEstFun = sample.totalReconstErrorEstFun;
				fd.totalReconstErrorEstHomography = sample.totalReconstErrorEstHomography;
				fd.totalReconstErrorEstEssential = sample.totalReconstErrorEstEssential;
				fd.medianReconstErrorEstFun = sample.medianReconstErrorEstFun;
				fd.medianReconstErrorEstHomography = sample.medianReconstErrorEstHomography;
				fd.medianReconstErrorEstEssential = sample.medianReconstErrorEstEssential;
				fd.transChordalEstFun = sample.transChordalEstFun;
				fd.transChordalEstHomography = sample.transChordalEstHomography;
				fd.transChordalEstEssential = sample.transChordalEstEssential;
				fd.baseline = 0;

				serializedData += fd.stringify();

				if (fd.totalReconstErrorEstHomography / fd.summary.numCorrespondences > 1
						|| fd.transChordalEstHomography > 0.4 || Double.isNaN(fd.totalReconstErrorEstHomography)) {
					rotationRejects++;
				} else {
					rotationAccepts++;
				}

			} else {
				rotationFails++;
				Utils.pl("FAILED");
			}

			Utils.pl("");
		}

		// pure rotation, noisy plane
		Utils.pl("");
		Utils.pl("=====================================================================================");
		Utils.pl("==========================  PLANAR ROTATION SCENE  ===========================");
		Utils.pl("=====================================================================================");
		Utils.pl("");
		int planarRotationIterations = 1000;
		rotRange = 0.25;
		rotOffset = rotRange / 2;
		VirtualEnvironment planarRotationMock = new VirtualEnvironment();
		planarRotationMock.generatePlanarScene(seed, 1000);

		for (int i = 0; i < planarRotationIterations; i++) {

			Utils.pl("iteration: " + i);

			// decide camera movement

			double rotX = rand.nextDouble() * rotRange - rotOffset;
			double rotY = rand.nextDouble() * rotRange - rotOffset;
			double rotZ = rand.nextDouble() * rotRange - rotOffset;

			planarRotationMock.getSecondaryCamera().setCx(0);
			planarRotationMock.getSecondaryCamera().setCy(0);
			planarRotationMock.getSecondaryCamera().setCz(0);
			planarRotationMock.getSecondaryCamera().setQw(1);
			planarRotationMock.getSecondaryCamera().setQx(0);
			planarRotationMock.getSecondaryCamera().setQy(0);
			planarRotationMock.getSecondaryCamera().setQz(0);
			planarRotationMock.getSecondaryCamera().rotateEuler(rotX, rotY, rotZ);

			Utils.pl("baseline: 0");
			Utils.pl("rotation:");
			Utils.pl(rotX);
			Utils.pl(rotY);
			Utils.pl(rotZ);

			Sample sample = new Sample();
			sample.evaluate(planarRotationMock);
			Utils.pl("numCorrespondences: " + sample.correspondences.size());

			if (sample.correspondences.size() >= 10) {
				FinalizedData fd = new FinalizedData();
				fd.summary = sample.correspondenceSummary;
				fd.totalReconstErrorEstFun = sample.totalReconstErrorEstFun;
				fd.totalReconstErrorEstHomography = sample.totalReconstErrorEstHomography;
				fd.totalReconstErrorEstEssential = sample.totalReconstErrorEstEssential;
				fd.medianReconstErrorEstFun = sample.medianReconstErrorEstFun;
				fd.medianReconstErrorEstHomography = sample.medianReconstErrorEstHomography;
				fd.medianReconstErrorEstEssential = sample.medianReconstErrorEstEssential;
				fd.transChordalEstFun = sample.transChordalEstFun;
				fd.transChordalEstHomography = sample.transChordalEstHomography;
				fd.transChordalEstEssential = sample.transChordalEstEssential;
				fd.baseline = 0;

				serializedData += fd.stringify();

				if (fd.totalReconstErrorEstHomography / fd.summary.numCorrespondences > 1
						|| fd.transChordalEstHomography > 0.4 || Double.isNaN(fd.totalReconstErrorEstHomography)) {
					rotationRejects++;
				} else {
					rotationAccepts++;
				}

			} else {
				rotationFails++;
				Utils.pl("FAILED");
			}

			Utils.pl("");
		}

		Utils.pl("rejects: " + rejects + " (" + (int) ((double) rejects * 100 / (rejects + accepts)) + "%)");
		Utils.pl("accepts: " + accepts + " (" + (int) ((double) accepts * 100 / (rejects + accepts)) + "%)");
		Utils.pl("failed: " + failed);
		Utils.pl("total: " + (rejects + accepts + failed));

		Utils.pl("");

		Utils.pl("rotation rejects: " + rotationRejects + " ("
				+ (int) ((double) rotationRejects * 100 / (rotationRejects + rotationAccepts)) + "%)");
		Utils.pl("rotation accepts: " + rotationAccepts + " ("
				+ (int) ((double) rotationAccepts * 100 / (rotationRejects + rotationAccepts)) + "%)");
		Utils.pl("rotation failed: " + rotationFails);
		Utils.pl("rotation total: " + (rotationRejects + rotationAccepts + rotationFails));

		// save data
		try {
			FileWriter fw = new FileWriter(OUT_FILE);
			fw.write(serializedData);
			fw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	public static List<FinalizedData> loadData(String filename) {
		List<FinalizedData> data = new ArrayList<FinalizedData>();

		try {

			BufferedReader br = new BufferedReader(new FileReader(filename));
			boolean keepGoing = true;
			while (keepGoing) {
				String dataLines = br.readLine();

				if (dataLines == null) {
					keepGoing = false;
					continue;
				}

				// ignore comment lines
				if (dataLines.trim().charAt(0) == '#') {
					continue;
				}

				dataLines += "\n" + br.readLine();

				FinalizedData fd = FinalizedData.parse(dataLines);
				data.add(fd);
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		return data;
	}

	public static void testModels() {

		String DATA_FILE = "results/data/TUM_samples_stn_3_30.dat";
//		String DATA_FILE = "results/data/testing-1617851095921-1615348975802.dat";

		// no-noise data models
//		String FUN_MODEL_FILE = "results/models/logreg_FUNDAMENTAL_10-03-2021_13-52-21_rotExcluded-P0.7967-R0.9071-F0.8483.model";
//		String ESS_MODEL_FILE = "results/models/logreg_ESSENTIAL_07-04-2021_23-06-55_rotExcluded-P0.8185-R0.7180-F0.7650.model";
//		String HOM_MODEL_FILE = "results/models/logreg_HOMOGRAPHY_10-03-2021_14-09-03_rotExcluded-P0.9086-R0.9739-F0.9401.model";
//		String ROT_MODEL_FILE = "results/models/logreg_ROTATION_10-03-2021_00-28-44-P0.7786-R0.9090-F0.8388.model";

		// noisy data models
//		String FUN_MODEL_FILE = "results/models/logreg_FUNDAMENTAL_27-04-2021_22-33-25_rotExcluded-P0.6698-R0.4441-F0.5341.model";
//		String ESS_MODEL_FILE = "results/models/logreg_ESSENTIAL_27-04-2021_22-39-32_rotExcluded-P0.7343-R0.3901-F0.5096.model";
//		String HOM_MODEL_FILE = "results/models/logreg_HOMOGRAPHY_27-04-2021_22-44-49_rotExcluded-P0.6970-R0.6122-F0.6519.model";
//		String ROT_MODEL_FILE = "results/models/logreg_ROTATION_27-04-2021_21-59-25-P0.8157-R0.8630-F0.8387.model";

//		// stf models ////
//		String FUN_MODEL_FILE = "results/models/logreg_real_FUNDAMENTAL_28-04-2021_15-30-09-P0.3639-R0.5822-F0.4479.model";
//		String ESS_MODEL_FILE = "results/models/logreg_real_ESSENTIAL_28-04-2021_15-35-38-P0.4290-R0.6453-F0.5154.model";
//		String HOM_MODEL_FILE = "results/models/logreg_real_HOMOGRAPHY_28-04-2021_15-40-33-P0.3725-R0.1166-F0.1776.model";
//		String ROT_MODEL_FILE = "results/models/logreg_real_ROTATION_28-04-2021_15-47-44-P0.8706-R0.7885-F0.8275.model";

//		// stn_2 models
//		String FUN_MODEL_FILE = "results/models/stn_train/logreg_real_FUNDAMENTAL_03-05-2021_21-40-51-P0.7943-R0.6619-F0.7221.model";
//		String ESS_MODEL_FILE = "results/models/stn_train/logreg_real_ESSENTIAL_03-05-2021_21-46-13-P0.7371-R0.7155-F0.7261.model";
//		String HOM_MODEL_FILE = "results/models/stn_train/logreg_real_HOMOGRAPHY_03-05-2021_21-51-32-P0.6512-R0.3944-F0.4912.model";
//		String ROT_MODEL_FILE = "results/models/stn_train/logreg_real_ROTATION_03-05-2021_21-56-54-P0.8800-R0.8527-F0.8661.model";

//		// stf_2 models
//		String FUN_MODEL_FILE = "results/models/stf_train/logreg_real_FUNDAMENTAL_04-05-2021_13-05-43-P0.7310-R0.7544-F0.7426.model";
//		String ESS_MODEL_FILE = "results/models/stf_train/logreg_real_ESSENTIAL_04-05-2021_13-12-13-P0.7452-R0.7175-F0.7311.model";
//		String HOM_MODEL_FILE = "results/models/stf_train/logreg_real_HOMOGRAPHY_04-05-2021_13-38-36-P0.6479-R0.3966-F0.4920.model";
//		String ROT_MODEL_FILE = "results/models/stf_train/logreg_real_ROTATION_04-05-2021_13-22-35-P0.9032-R0.6222-F0.7368.model";

		// updated models (ORBSLAM labeling)
		String FUN_MODEL_FILE = "results/models/updated_models_sept_15/logreg_real_FUNDAMENTAL_13-09-2021_15-06-08-P0.5434-R0.7797-F0.6405.model";
		String ESS_MODEL_FILE = "results/models/updated_models_sept_15/logreg_real_ESSENTIAL_13-09-2021_15-14-57-P0.6239-R0.8454-F0.7179.model";
		String HOM_MODEL_FILE = "results/models/updated_models_sept_15/logreg_real_HOMOGRAPHY_13-09-2021_15-20-42-P0.4492-R0.4543-F0.4517.model";

		String ROT_MODEL_FILE = "results/models/stf_train/logreg_real_ROTATION_04-05-2021_13-22-35-P0.9032-R0.6222-F0.7368.model";

		// get the models
		MultiLayerNetwork modelFun = null;
		MultiLayerNetwork modelEss = null;
		MultiLayerNetwork modelHom = null;
		MultiLayerNetwork modelRot = null;

		try {
			modelFun = MultiLayerNetwork.load(new File(FUN_MODEL_FILE), true);
			modelEss = MultiLayerNetwork.load(new File(ESS_MODEL_FILE), true);
			modelHom = MultiLayerNetwork.load(new File(HOM_MODEL_FILE), true);
			modelRot = MultiLayerNetwork.load(new File(ROT_MODEL_FILE), true);
		} catch (Exception e) {
			e.printStackTrace();
		}

		// testing
		Map<String, INDArray> paramsFun = modelFun.paramTable();
		Utils.pl("PARAMS fun");
		Utils.pl(paramsFun);

		Map<String, INDArray> paramsEss = modelEss.paramTable();
		Utils.pl("PARAMS ess");
		Utils.pl(paramsEss);

		Map<String, INDArray> paramsHom = modelHom.paramTable();
		Utils.pl("PARAMS hom");
		Utils.pl(paramsHom);

		// get the data
		List<FinalizedData> data = loadData(DATA_FILE);

		// get predictions
		double[] predFun = getPredictions(modelFun, data);
		double[] predEss = getPredictions(modelEss, data);
		double[] predHom = getPredictions(modelHom, data);
		double[] predRot = getPredictions(modelRot, data);

		// establish index list
		double[] indices = new double[data.size()];
		for (int i = 0; i < indices.length; i++) {
			indices[i] = i + 1;
		}

		// chart results
		final XYChart chart = new XYChartBuilder().width(640).height(480).theme(Styler.ChartTheme.Matlab)
				.title("Predictions on Batches (>0.5 means positive class)").xAxisTitle("Frame Number")
				.yAxisTitle("Prediction").build();

		// Customize Chart
		chart.getStyler().setLegendPosition(LegendPosition.InsideNE);

		// Series
		chart.addSeries("Fundamental Matrix Estimate (8PA)", indices, predFun);
		chart.addSeries("Essential Matrix Estimate (5PA)", indices, predEss);
		chart.addSeries("Homography Estimate (4PA)", indices, predHom);
		chart.addSeries("Pure Rotation", indices, predRot);

		// Show it
		new SwingWrapper(chart).displayChart();

		// evaulate data
		LabelMaker labelerF = new LabelMaker() {
			public int getLabel(FinalizedData fd) {
				return getLabelFundamental(fd);
			}
		};
		LabelMaker labelerE = new LabelMaker() {
			public int getLabel(FinalizedData fd) {
				return getLabelEssential(fd);
			}
		};
		LabelMaker labelerH = new LabelMaker() {
			public int getLabel(FinalizedData fd) {
				return getLabelHomography(fd);
			}
		};
		LabelMaker labelerR = new LabelMaker() {
			public int getLabel(FinalizedData fd) {
				return getLabelRotation(fd);
			}
		};

		Utils.pl("+++++++++++++++++++++++++    FUNDAMENTAL    +++++++++++++++++++++++++");
		evaluateData(DATA_FILE, modelFun, labelerF, false);
		Utils.pl("+++++++++++++++++++++++++     ESSENTIAL     +++++++++++++++++++++++++");
		evaluateData(DATA_FILE, modelEss, labelerE, false);
		Utils.pl("+++++++++++++++++++++++++     HOMOGRAPHY    +++++++++++++++++++++++++");
		evaluateData(DATA_FILE, modelHom, labelerH, false);
		Utils.pl("+++++++++++++++++++++++++     ROTATION      +++++++++++++++++++++++++");
		evaluateData(DATA_FILE, modelRot, labelerR, false);

		Utils.pl("////////////  BUTT  ///////////////");
		Utils.pl("fundamental: ");
		evaluateButt(DATA_FILE, MODE.FUNDAMENTAL, labelerF, false);
		Utils.pl("homography: ");
		evaluateButt(DATA_FILE, MODE.HOMOGRAPHY, labelerH, false);

		// evaluate models with pure rotation model filter
		Utils.pl("------------------    FUNDAMENTAL FILTERED    -------------------");
		evaluateDataFiltered(DATA_FILE, modelFun, modelRot, labelerF, false);
		Utils.pl("------------------    ESSENTIAL FILTERED    -------------------");
		evaluateDataFiltered(DATA_FILE, modelEss, modelRot, labelerE, false);
		Utils.pl("------------------    HOMOGRAPHY FILTERED    -------------------");
		evaluateDataFiltered(DATA_FILE, modelHom, modelRot, labelerH, false);

	}

	public static void evaluateData(String testingFilename, MultiLayerNetwork model, LabelMaker labeler,
			boolean excludePureRotation) {
		// load testing data
		List<FinalizedData> pretest = loadData(testingFilename);
		List<FinalizedData> test = new ArrayList<FinalizedData>();
		for (int i = 0; i < pretest.size(); i++) {
			if (!excludePureRotation || pretest.get(i).baseline > 0) {
				test.add(pretest.get(i));
			}
		}

		// generate test labels
		INDArray testLabels = Nd4j.zeros(test.size(), 1);
		for (int i = 0; i < test.size(); i++) {
			testLabels.putScalar(new int[] { i, 0 }, labeler.getLabel(test.get(i)));
		}

		// create input array (normalize data)
		INDArray testInput = Nd4j.zeros(test.size(), 23);
		for (int i = 0; i < test.size(); i++) {
			INDArray row = Nd4j.create(test.get(i).summary.getArray());
			testInput.putRow(i, row);
		}

		// set up test data
		DataSet testData = new DataSet(testInput, testLabels);

		// create output for every training sample
		long start = System.currentTimeMillis();
		INDArray output = model.output(testData.getFeatures());
		long end = System.currentTimeMillis();
		long totalTime = end - start;
		Utils.pl("Total prediction time: " + totalTime + "ms");
		Utils.pl("Number of rows: " + testInput.rows());
		Utils.pl("Avg prediction time: " + (totalTime / (double) testInput.rows()) + "ms");

//		System.out.println(output);

		// let Evaluation prints stats how often the right output had the highest value
		Evaluation eval = new Evaluation();
		eval.eval(testData.getLabels(), output);
		System.out.println(eval.stats());
	}

	public static void evaluateDataManual(String testingFilename, MultiLayerNetwork model, LabelMaker labeler,
			boolean excludePureRotation) {
		// load testing data
		List<FinalizedData> pretest = loadData(testingFilename);
		List<FinalizedData> test = new ArrayList<FinalizedData>();
		for (int i = 0; i < pretest.size(); i++) {
			if (!excludePureRotation || pretest.get(i).baseline > 0) {
				test.add(pretest.get(i));
			}
		}

		// generate test labels
		INDArray testLabels = Nd4j.zeros(test.size(), 1);
		for (int i = 0; i < test.size(); i++) {
			testLabels.putScalar(new int[] { i, 0 }, labeler.getLabel(test.get(i)));
		}

		// create input array (normalize data)
		INDArray testInput = Nd4j.zeros(test.size(), 23);
		for (int i = 0; i < test.size(); i++) {
			INDArray row = Nd4j.create(test.get(i).summary.getArray());
			testInput.putRow(i, row);
		}

		// set up test data
		DataSet testData = new DataSet(testInput, testLabels);

		// create output for every training sample
		INDArray output = model.output(testData.getFeatures());
//		System.out.println(output);

		// let Evaluation prints stats how often the right output had the highest value
//		Evaluation eval = new Evaluation();
//		eval.eval(testData.getLabels(), output);
//		System.out.println(eval.stats());
		double tp = 0;
		double tn = 0;
		double fp = 0;
		double fn = 0;

		for (int i = 0; i < output.rows(); i++) {
			int predicted = (int) Math.round(output.getDouble(i));
			int actual = (int) testData.getLabels().getDouble(i);

			if (predicted == actual) {
				if (predicted == 1) {
					tp++;
				} else {
					tn++;
				}
			} else {
				if (predicted == 1) {
					fp++;
				} else {
					fn++;
				}
			}
		}

		Utils.pl("tp: " + tp);
		Utils.pl("tn: " + tn);
		Utils.pl("fp: " + fp);
		Utils.pl("fn: " + fn);

		double accuracy = (tp + tn) / (tp + tn + fp + fn);
		double precision = (tp + fp) == 0 ? 0 : tp / (tp + fp);
		double recall = (tp + fn) == 0 ? 0 : tp / (tp + fn);
		double f1 = (tp + 0.5 * (fp + fn)) == 0 ? 0 : tp / (tp + 0.5 * (fp + fn));

		Utils.pl("Accuracy: " + accuracy);
		Utils.pl("Precision: " + precision);
		Utils.pl("Recall: " + recall);
		Utils.pl("F1: " + f1);

	}

	public static void evaluateDataFiltered(String testingFilename, MultiLayerNetwork model, MultiLayerNetwork rotModel,
			LabelMaker labeler, boolean excludePureRotation) {
		// load testing data
		List<FinalizedData> pretest = loadData(testingFilename);
		List<FinalizedData> test = new ArrayList<FinalizedData>();
		for (int i = 0; i < pretest.size(); i++) {
			if (!excludePureRotation || pretest.get(i).baseline > 0) {
				test.add(pretest.get(i));
			}
		}

		// generate test labels
		INDArray testLabels = Nd4j.zeros(test.size(), 1);
		for (int i = 0; i < test.size(); i++) {
			testLabels.putScalar(new int[] { i, 0 }, labeler.getLabel(test.get(i)));
		}

		// create input array (normalize data)
		INDArray testInput = Nd4j.zeros(test.size(), 23);
		for (int i = 0; i < test.size(); i++) {
			INDArray row = Nd4j.create(test.get(i).summary.getArray());
			testInput.putRow(i, row);
		}

		// set up test data
		DataSet testData = new DataSet(testInput, testLabels);

		// create output for every test sample
		INDArray output = model.output(testData.getFeatures());
		INDArray rotOutput = rotModel.output(testData.getFeatures());
//		System.out.println(output);

		// let Evaluation prints stats how often the right output had the highest value
//		Evaluation eval = new Evaluation();
//		eval.eval(testData.getLabels(), output);
//		System.out.println(eval.stats());
		double tp = 0;
		double tn = 0;
		double fp = 0;
		double fn = 0;

		for (int i = 0; i < output.rows(); i++) {
			int predRot = (int) Math.round(rotOutput.getDouble(i));
			int predicted = predRot == 0 ? (int) Math.round(output.getDouble(i)) : 0;
			int actual = (int) testData.getLabels().getDouble(i);

			if (predicted == actual) {
				if (predicted == 1) {
					tp++;
				} else {
					tn++;
				}
			} else {
				if (predicted == 1) {
					fp++;
				} else {
					fn++;
				}
			}
		}

		Utils.pl("tp: " + tp);
		Utils.pl("tn: " + tn);
		Utils.pl("fp: " + fp);
		Utils.pl("fn: " + fn);

		double accuracy = (tp + tn) / (tp + tn + fp + fn);
		double precision = (tp + fp) == 0 ? 0 : tp / (tp + fp);
		double recall = (tp + fn) == 0 ? 0 : tp / (tp + fn);
		double f1 = (tp + 0.5 * (fp + fn)) == 0 ? 0 : tp / (tp + 0.5 * (fp + fn));

		Utils.pl("Accuracy: " + accuracy);
		Utils.pl("Precision: " + precision);
		Utils.pl("Recall: " + recall);
		Utils.pl("F1: " + f1);

	}

	public static void evaluateButt(String testingFilename, MODE mode, LabelMaker labeler,
			boolean excludePureRotation) {

		// load testing data
		List<FinalizedData> pretest = loadData(testingFilename);
		List<FinalizedData> test = new ArrayList<FinalizedData>();
		for (int i = 0; i < pretest.size(); i++) {
			if (!excludePureRotation || pretest.get(i).baseline > 0) {
				test.add(pretest.get(i));
			}
		}

		// generate test labels
		INDArray testLabels = Nd4j.zeros(test.size(), 1);
		for (int i = 0; i < test.size(); i++) {
			testLabels.putScalar(new int[] { i, 0 }, labeler.getLabel(test.get(i)));
		}

		double tp = 0;
		double tn = 0;
		double fp = 0;
		double fn = 0;

		for (int i = 0; i < test.size(); i++) {
			int predicted = test.get(i).summary.meanDisparity > 50 ? 1 : 0;
			int actual = (int) testLabels.getDouble(i);

			if (mode == MODE.FUNDAMENTAL && predicted == 1) {
				predicted = test.get(i).summary.stdDevDisparity > 15 ? 1 : 0;
			}

			if (predicted == actual) {
				if (predicted == 1) {
					tp++;
				} else {
					tn++;
				}
			} else {
				if (predicted == 1) {
					fp++;
				} else {
					fn++;
				}
			}
		}

		Utils.pl("tp: " + tp);
		Utils.pl("tn: " + tn);
		Utils.pl("fp: " + fp);
		Utils.pl("fn: " + fn);

		double accuracy = (tp + tn) / (tp + tn + fp + fn);
		double precision = (tp + fp) == 0 ? 0 : tp / (tp + fp);
		double recall = (tp + fn) == 0 ? 0 : tp / (tp + fn);
		double f1 = (tp + 0.5 * (fp + fn)) == 0 ? 0 : tp / (tp + 0.5 * (fp + fn));

		Utils.pl("Accuracy: " + accuracy);
		Utils.pl("Precision: " + precision);
		Utils.pl("Recall: " + recall);
		Utils.pl("F1: " + f1);

	}

	public static double[] getPredictions(MultiLayerNetwork model, List<FinalizedData> data) {

		// create input array (normalize data)
		Utils.pl("row:");
		INDArray input = Nd4j.zeros(data.size(), 23);
		for (int i = 0; i < data.size(); i++) {
			INDArray row = Nd4j.create(data.get(i).summary.getArray());
			if (i == 0) {
				Utils.pl(row);
			}

			input.putRow(i, row);
		}

		INDArray output = model.output(input);

		Utils.pl("prediction:");
		double[] predictions = new double[output.rows()];
		for (int i = 0; i < output.rows(); i++) {
			if (i == 0) {
				Utils.pl(output.getDouble(i));
			}
			predictions[i] = output.getDouble(i);
		}

		return predictions;

	}

}
