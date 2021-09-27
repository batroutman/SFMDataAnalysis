import java.util.ArrayList;
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

import Jama.Matrix;

public class Sample {

	///////////////////////////////////////////////////////////////////////////////
	/////////////////////////// GIVEN INFORMATION /////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////

	public Pose primaryCamera = null;
	public Pose secondaryCamera = null;
	public List<Correspondence2D2D> correspondences = new ArrayList<Correspondence2D2D>();
	public List<Matrix> truePoints = new ArrayList<Matrix>();
	public Matrix trueFundamentalMatrix = null;

	///////////////////////////////////////////////////////////////////////////////
	///////////////////////// CORRESPONDENCE SUMMARY //////////////////////////////
	///////////////////////////////////////////////////////////////////////////////
	public CorrespondenceSummary correspondenceSummary = new CorrespondenceSummary();

	///////////////////////////////////////////////////////////////////////////////
	/////////////////////////// ITEMS TO ESTIMATE /////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////

	public Matrix estimatedFundamentalMatrix = null;
	public Mat estimatedHomography = null;
	public Matrix estimatedEssentialMatrix = null;

	///////////////////////////////////////////////////////////////////////////////
	/////////////////////// SECONDARY POSES TO ESTIMATE ///////////////////////////
	///////////////////////////////////////////////////////////////////////////////

	// pose derived from true fundamental matrix
	public Matrix poseTrueFun = null;

	// pose derived from estimated fundamental matrix
	public Matrix poseEstFun = null;

	// pose derived from homography
	public Matrix poseEstHomography = null;

	// debug cheirality poses from homgraphy
	public List<Matrix> homCheiralityPoses = new ArrayList<Matrix>();

	// pose derived from estimated essential matrix
	public Matrix poseEstEssential = null;

	///////////////////////////////////////////////////////////////////////////////
	///////////////////////// 3D POINT ESTIMATIONS ////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////

	// points triangulated from secondaryCamera
	public List<Matrix> estPointsTrue = new ArrayList<Matrix>();

	// points triangulated from poseTrueFun
	public List<Matrix> estPointsTrueFun = new ArrayList<Matrix>();

	// points triangulated from poseEstFun
	public List<Matrix> estPointsEstFun = new ArrayList<Matrix>();

	// points triangulated from poseEstHomography
	public List<Matrix> estPointsEstHomography = new ArrayList<Matrix>();

	// points triangulated from poseEstEssential
	public List<Matrix> estPointsEstEssential = new ArrayList<Matrix>();

	///////////////////////////////////////////////////////////////////////////////
	///////////////////////// RECONSTRUCTION ERRORS ///////////////////////////////
	///////////////////////////////////////////////////////////////////////////////

	// sum of euclidean distances between true 3D points and true triangulated
	// points
	double totalReconstErrorTrue = 0;
	double medianReconstErrorTrue = 0;

	// sum of euclidean distances between true 3D points and true fundamental matrix
	// triangulated points
	double totalReconstErrorTrueFun = 0;
	double medianReconstErrorTrueFun = 0;

	// sum of euclidean distances between true 3D points and estimated fundamental
	// matrix triangulated points
	double totalReconstErrorEstFun = 0;
	double medianReconstErrorEstFun = 0;

	// sum of euclidean distances between true 3D points and estimated homography
	// triangulated points
	double totalReconstErrorEstHomography = 0;
	double medianReconstErrorEstHomography = 0;

	// sum of euclidean distances between true 3D points and estimated essential
	// matrix triangulated points
	double totalReconstErrorEstEssential = 0;
	double medianReconstErrorEstEssential = 0;

	///////////////////////////////////////////////////////////////////////////////
	////////////////////////// REPROJECTION ERRORS ////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////

	// sum of reprojection errors using primaryCamera, secondaryCamera, and
	// estPointsTrue
	public double totalReprojErrorTrue = 0;

	// sum of reprojection errors using primaryCamera, poseTrueFun, and
	// estPointsTrueFun
	public double totalReprojErrorTrueFun = 0;

	// sum of reprojection errors using primaryCamera, poseEstFun, and
	// estPointsEstFun
	public double totalReprojErrorEstFun = 0;

	// sum of reprojection errors using primaryCamera, poseEstHomography, and
	// estPointsEstHomography
	public double totalReprojErrorEstHomography = 0;

	// sum of reprojection errors using primaryCamera, poseEstEssential, and
	// estPointsEstEssential
	public double totalReprojErrorEstEssential = 0;

	///////////////////////////////////////////////////////////////////////////////
	/////////////////////////// CHORDAL DISTANCES /////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////

	// ---- rotational chordal distances ----
	public double rotChordalTrueFun = 0;
	public double rotChordalEstFun = 0;
	public double rotChordalEstHomography = 0;
	public double rotChordalEstEssential = 0;

	// -- normalized translational chordal distances ----
	public double transChordalTrueFun = 0;
	public double transChordalEstFun = 0;
	public double transChordalEstHomography = 0;
	public double transChordalEstEssential = 0;

	///////////////////////////////////////////////////////////////////////////////
	//////////////////////// RECONSTRUCTION CRITERIA //////////////////////////////
	///////////////////////////////////////////////////////////////////////////////
	public double funNumGood = 0;
	public double funNumParallax = 0;
	public double essNumGood = 0;
	public double essNumParallax = 0;
	public double homNumGood = 0;
	public double homNumParallax = 0;

	public Sample() {
	}

	public void evaluate(VirtualEnvironment mock) {
		evaluate(mock.getPrimaryCamera(), mock.getSecondaryCamera(), mock.getCorrespondences(this.truePoints),
				mock.getCameraParams(), mock.getTrueFundamentalMatrix(), false);
	}

	public void evaluate(Pose primaryCamera, Pose secondaryCamera, List<Correspondence2D2D> correspondences,
			CameraParams cameraParams, Matrix trueFunMat, boolean calcTruePoints) {

		// set cameras
		this.primaryCamera = primaryCamera;
		this.secondaryCamera = secondaryCamera;

		// get correspondences, worldPoints, and true fundamental matrix
		this.correspondences = correspondences;
		this.trueFundamentalMatrix = trueFunMat;

		if (calcTruePoints) {
			this.truePoints.clear();
			for (int i = 0; i < correspondences.size(); i++) {
				this.truePoints.add(ComputerVision.triangulate(secondaryCamera.getHomogeneousMatrix(),
						primaryCamera.getHomogeneousMatrix(), cameraParams, correspondences.get(i)));
			}
		}

		// correspondence summary
		this.correspondenceSummary.evaluate(this.correspondences);

		// if not enough correspondences, terminate now
		if (this.correspondences.size() < 10) {
			return;
		}

		// estimated fundamental matrix and homography
		this.estimatedFundamentalMatrix = ComputerVision.estimateFundamentalMatrix(this.correspondences);
		this.estimatedHomography = ComputerVision.estimateHomography(this.correspondences);
		this.estimatedEssentialMatrix = ComputerVision.estimateEssentialMatrix(this.correspondences, cameraParams);

		// estimated poses
		this.poseTrueFun = ComputerVision.getPoseFromFundamentalMatrix(this.trueFundamentalMatrix, cameraParams,
				this.correspondences);
		this.poseEstFun = ComputerVision.getPoseFromFundamentalMatrix(this.estimatedFundamentalMatrix, cameraParams,
				this.correspondences);
		try {
			this.poseEstHomography = ComputerVision.getPoseFromHomography(this.estimatedHomography, this.primaryCamera,
					cameraParams, correspondences, homCheiralityPoses);
		} catch (Exception e) {
			Utils.pl("Homography estimation broke. Defaulting to identity.");
			this.poseEstHomography = Matrix.identity(4, 4);
		}

		this.poseEstEssential = ComputerVision.getPoseFromEssentialMatrix(this.estimatedEssentialMatrix, cameraParams,
				this.correspondences);

		// 3D point estimations
		this.estPointsTrue = ComputerVision.triangulateCorrespondences(this.secondaryCamera.getHomogeneousMatrix(),
				this.primaryCamera.getHomogeneousMatrix(), cameraParams, correspondences);
		this.estPointsTrueFun = ComputerVision.triangulateCorrespondences(poseTrueFun,
				this.primaryCamera.getHomogeneousMatrix(), cameraParams, correspondences);
		this.estPointsEstFun = ComputerVision.triangulateCorrespondences(poseEstFun,
				this.primaryCamera.getHomogeneousMatrix(), cameraParams, correspondences);
		this.estPointsEstHomography = ComputerVision.triangulateCorrespondences(poseEstHomography,
				this.primaryCamera.getHomogeneousMatrix(), cameraParams, correspondences);
		this.estPointsEstEssential = ComputerVision.triangulateCorrespondences(poseEstEssential,
				this.primaryCamera.getHomogeneousMatrix(), cameraParams, correspondences);

		// calculate error metrics
		this.errorMetrics();

		// robust reconstruction metrics
		Dbl funParallax = new Dbl(0);
		Dbl funGood = new Dbl(0);
		Dbl homParallax = new Dbl(0);
		Dbl homGood = new Dbl(0);
		Dbl essParallax = new Dbl(0);
		Dbl essGood = new Dbl(0);

		ComputerVision.parallaxAndGoodPoints(this.primaryCamera.getHomogeneousMatrix(), this.poseEstFun, cameraParams,
				this.estPointsEstFun, this.correspondences, funParallax, funGood);
		ComputerVision.parallaxAndGoodPoints(this.primaryCamera.getHomogeneousMatrix(), this.poseEstHomography,
				cameraParams, this.estPointsEstHomography, this.correspondences, homParallax, homGood);
		ComputerVision.parallaxAndGoodPoints(this.primaryCamera.getHomogeneousMatrix(), this.poseEstEssential,
				cameraParams, this.estPointsEstEssential, this.correspondences, essParallax, essGood);

		this.funNumGood = funGood.getValue();
		this.funNumParallax = funParallax.getValue();
		this.homNumGood = homGood.getValue();
		this.homNumParallax = homParallax.getValue();
		this.essNumGood = essGood.getValue();
		this.essNumParallax = essParallax.getValue();

	}

	public void printErrors() {
		Utils.pl("Total reprojection error (true pose): " + this.totalReprojErrorTrue);
		Utils.pl("Total reprojection error (true fundamental matrix): " + this.totalReprojErrorTrueFun);
		Utils.pl("Total reprojection error (estimated fundamental matrix): " + this.totalReprojErrorEstFun);
		Utils.pl("Total reprojection error (estimated homography): " + this.totalReprojErrorEstHomography);
		Utils.pl("Total reprojection error (estimated essential matrix): " + this.totalReprojErrorEstEssential);

		Utils.pl(
				"Average reprojection error (true pose): " + (this.totalReprojErrorTrue / this.correspondences.size()));
		Utils.pl("Average reprojection error (true fundamental matrix): "
				+ (this.totalReprojErrorTrueFun / this.correspondences.size()));
		Utils.pl("Average reprojection error (estimated fundamental matrix): "
				+ (this.totalReprojErrorEstFun / this.correspondences.size()));
		Utils.pl("Average reprojection error (estimated homography): "
				+ (this.totalReprojErrorEstHomography / this.correspondences.size()));
		Utils.pl("Average reprojection error (estimated essential matrix): "
				+ (this.totalReprojErrorEstEssential / this.correspondences.size()));

		Utils.pl("Rotational chordal distance (true fundamental matrix): " + this.rotChordalTrueFun);
		Utils.pl("Rotational chordal distance (estimated fundamental matrix): " + this.rotChordalEstFun);
		Utils.pl("Rotational chordal distance (estimated homography): " + this.rotChordalEstHomography);
		Utils.pl("Rotational chordal distance (estimated essential matrix): " + this.rotChordalEstEssential);

		Utils.pl("Normalized translational chordal distance (true fundamental matrix): " + this.transChordalTrueFun);
		Utils.pl(
				"Normalized translational chordal distance (estimated fundamental matrix): " + this.transChordalEstFun);
		Utils.pl("Normalized translational chordal distance (estimated homography): " + this.transChordalEstHomography);
		Utils.pl("Normalized translational chordal distance (estimated essential matrix): "
				+ this.transChordalEstEssential);
	}

	public String stringify() {
		String output = "";

		// cameras (qw,qx,qy,qz,Cx,Cy,Cz)
		Pose cam1 = this.primaryCamera;
		Pose cam2 = this.secondaryCamera;
		output += cam1.getQw() + "," + cam1.getQx() + "," + cam1.getQy() + "," + cam1.getQz() + "," + cam1.getCx() + ","
				+ cam1.getCy() + "," + cam1.getCz() + "\n";
		output += cam2.getQw() + "," + cam2.getQx() + "," + cam2.getQy() + "," + cam2.getQz() + "," + cam2.getCx() + ","
				+ cam2.getCy() + "," + cam2.getCz() + "\n";

		// correspondences
		// number of correspondences
		output += this.correspondences.size() + "\n";

		// correspondences (x0,y0,x1,y1|x0,y0,x1,y1|x0,y0,x1,y1|...)
		for (int i = 0; i < this.correspondences.size(); i++) {
			Correspondence2D2D c = this.correspondences.get(i);
			output += c.getX0() + "," + c.getY0() + "," + c.getX1() + "," + c.getY1() + "|";
		}
		output += "\n";

		// correspondence summary
		output += this.correspondenceSummary.stringify();

		// true points
		for (int i = 0; i < this.truePoints.size(); i++) {
			Matrix p = this.truePoints.get(i);
			output += p.get(0, 0) + "," + p.get(1, 0) + "," + p.get(2, 0) + "|";
		}
		output += "\n";

		// true fundamental matrix (row major order)
		for (int i = 0; i < this.trueFundamentalMatrix.getRowDimension(); i++) {
			for (int j = 0; j < this.trueFundamentalMatrix.getColumnDimension(); j++) {
				output += this.trueFundamentalMatrix.get(i, j) + ",";
			}
		}
		output += "\n";

		// estimated fundamental matrix (row major order)
		for (int i = 0; i < this.estimatedFundamentalMatrix.getRowDimension(); i++) {
			for (int j = 0; j < this.estimatedFundamentalMatrix.getColumnDimension(); j++) {
				output += this.estimatedFundamentalMatrix.get(i, j) + ",";
			}
		}
		output += "\n";

		// estimated homography (row major order)
		for (int i = 0; i < this.estimatedHomography.rows(); i++) {
			for (int j = 0; j < this.estimatedHomography.cols(); j++) {
				output += this.estimatedHomography.get(i, j)[0] + ",";
			}
		}
		output += "\n";

		// estimated essential matrix (row major order)
		for (int i = 0; i < this.estimatedEssentialMatrix.getRowDimension(); i++) {
			for (int j = 0; j < this.estimatedEssentialMatrix.getColumnDimension(); j++) {
				output += this.estimatedEssentialMatrix.get(i, j) + ",";
			}
		}
		output += "\n";

		// poseTrueFun matrix (row major order)
		for (int i = 0; i < this.poseTrueFun.getRowDimension(); i++) {
			for (int j = 0; j < this.poseTrueFun.getColumnDimension(); j++) {
				output += this.poseTrueFun.get(i, j) + ",";
			}
		}
		output += "\n";

		// poseEstFun matrix (row major order)
		for (int i = 0; i < this.poseEstFun.getRowDimension(); i++) {
			for (int j = 0; j < this.poseEstFun.getColumnDimension(); j++) {
				output += this.poseEstFun.get(i, j) + ",";
			}
		}
		output += "\n";

		// poseEstHomography matrix (row major order)
		for (int i = 0; i < this.poseEstHomography.getRowDimension(); i++) {
			for (int j = 0; j < this.poseEstHomography.getColumnDimension(); j++) {
				output += this.poseEstHomography.get(i, j) + ",";
			}
		}
		output += "\n";

		// poseEstEssential matrix (row major order)
		for (int i = 0; i < this.poseEstEssential.getRowDimension(); i++) {
			for (int j = 0; j < this.poseEstEssential.getColumnDimension(); j++) {
				output += this.poseEstEssential.get(i, j) + ",";
			}
		}
		output += "\n";

		// estPointsTrue (x,y,z|x,y,z|...)
		for (int i = 0; i < this.estPointsTrue.size(); i++) {
			Matrix p = this.estPointsTrue.get(i);
			output += p.get(0, 0) + "," + p.get(1, 0) + "," + p.get(2, 0) + "|";
		}
		output += "\n";

		// estPointsTrueFun (x,y,z|x,y,z|...)
		for (int i = 0; i < this.estPointsTrueFun.size(); i++) {
			Matrix p = this.estPointsTrueFun.get(i);
			output += p.get(0, 0) + "," + p.get(1, 0) + "," + p.get(2, 0) + "|";
		}
		output += "\n";

		// estPointsEstFun (x,y,z|x,y,z|...)
		for (int i = 0; i < this.estPointsEstFun.size(); i++) {
			Matrix p = this.estPointsEstFun.get(i);
			output += p.get(0, 0) + "," + p.get(1, 0) + "," + p.get(2, 0) + "|";
		}
		output += "\n";

		// estPointsEstHomography (x,y,z|x,y,z|...)
		for (int i = 0; i < this.estPointsEstHomography.size(); i++) {
			Matrix p = this.estPointsEstHomography.get(i);
			output += p.get(0, 0) + "," + p.get(1, 0) + "," + p.get(2, 0) + "|";
		}
		output += "\n";

		// estPointsEstEssential (x,y,z|x,y,z|...)
		for (int i = 0; i < this.estPointsEstEssential.size(); i++) {
			Matrix p = this.estPointsEstEssential.get(i);
			output += p.get(0, 0) + "," + p.get(1, 0) + "," + p.get(2, 0) + "|";
		}
		output += "\n";

		// total reconstruction errors
		// (totalReconstErrorTrue,totalReconstErrorTrueFun,totalReconstErrorEstFun,totalReconstErrorEstHomography)
		output += this.totalReconstErrorTrue + "," + this.totalReconstErrorTrueFun + "," + this.totalReconstErrorEstFun
				+ "," + this.totalReconstErrorEstHomography + "," + this.totalReconstErrorEstEssential + "\n";

		// median reconstruction errors
		// (medianReconstErrorTrue,medianReconstErrorTrueFun,medianReconstErrorEstFun,medianReconstErrorEstHomography)
		output += this.medianReconstErrorTrue + "," + this.medianReconstErrorTrueFun + ","
				+ this.medianReconstErrorEstFun + "," + this.medianReconstErrorEstHomography + ","
				+ this.medianReconstErrorEstEssential + "\n";

		// total reprojection errors
		// (totalReprojErrorTrue,totalReprojErrorTrueFun,totalReprojErrorEstFun,totalReprojErrorEstHomography)
		output += this.totalReprojErrorTrue + "," + this.totalReprojErrorTrueFun + "," + this.totalReprojErrorEstFun
				+ "," + this.totalReprojErrorEstHomography + "," + this.totalReprojErrorEstEssential + "\n";

		// rotational chordal distances
		// (rotChordalTrueFun,rotChordalEstFun,rotChordalEstHomography)
		output += this.rotChordalTrueFun + "," + this.rotChordalEstFun + "," + this.rotChordalEstHomography + ","
				+ this.rotChordalEstEssential + "\n";

		// translational chordal distances
		// (transChordalTrueFun,transChordalEstFun,transChordalEstHomography)
		output += this.transChordalTrueFun + "," + this.transChordalEstFun + "," + this.transChordalEstHomography + ","
				+ this.transChordalEstEssential + "\n";

		return output;
	}

	public static Sample parse(String input) {
		Sample sample = new Sample();

		String[] lines = input.split("\n");
		int line = 0;

		// cameras
		// camera 1
		String[] cam1Line = lines[line].split(",");
		Pose camera1 = new Pose();
		camera1.setQw(Double.parseDouble(cam1Line[0]));
		camera1.setQx(Double.parseDouble(cam1Line[1]));
		camera1.setQy(Double.parseDouble(cam1Line[2]));
		camera1.setQz(Double.parseDouble(cam1Line[3]));
		camera1.setCx(Double.parseDouble(cam1Line[4]));
		camera1.setCy(Double.parseDouble(cam1Line[5]));
		camera1.setCz(Double.parseDouble(cam1Line[6]));
		sample.primaryCamera = camera1;
		line++;

		// camera 2
		String[] cam2Line = lines[line].split(",");
		Pose camera2 = new Pose();
		camera2.setQw(Double.parseDouble(cam2Line[0]));
		camera2.setQx(Double.parseDouble(cam2Line[1]));
		camera2.setQy(Double.parseDouble(cam2Line[2]));
		camera2.setQz(Double.parseDouble(cam2Line[3]));
		camera2.setCx(Double.parseDouble(cam2Line[4]));
		camera2.setCy(Double.parseDouble(cam2Line[5]));
		camera2.setCz(Double.parseDouble(cam2Line[6]));
		sample.secondaryCamera = camera2;
		line++;

		// correspondences
		int numCorrespondences = Integer.parseInt(lines[line]);
		List<Correspondence2D2D> correspondences = new ArrayList<Correspondence2D2D>(numCorrespondences);
		line++;

		String[] correspondenceLine = lines[line].split("\\|");
		for (int i = 0; i < correspondenceLine.length; i++) {
			String[] corr = correspondenceLine[i].split(",");
			Correspondence2D2D c = new Correspondence2D2D();
			c.setX0(Double.parseDouble(corr[0]));
			c.setY0(Double.parseDouble(corr[1]));
			c.setX1(Double.parseDouble(corr[2]));
			c.setY1(Double.parseDouble(corr[3]));
			correspondences.add(c);
		}
		sample.correspondences = correspondences;
		line++;

		// correspondence summary
		sample.correspondenceSummary = CorrespondenceSummary.parse(lines[line]);
		line++;

		// estPointsTrue
		String[] truePointsLine = lines[line].split("\\|");
		List<Matrix> truePoints = new ArrayList<Matrix>(truePointsLine.length);
		for (int i = 0; i < truePointsLine.length; i++) {
			String[] coords = truePointsLine[i].split(",");
			Matrix p = new Matrix(4, 1);
			p.set(0, 0, Double.parseDouble(coords[0]));
			p.set(1, 0, Double.parseDouble(coords[1]));
			p.set(2, 0, Double.parseDouble(coords[2]));
			p.set(3, 0, 1);
			truePoints.add(p);
		}
		sample.truePoints = truePoints;
		line++;

		// true fundamental matrix
		String[] trueFunMatLine = lines[line].split(",");
		Matrix trueFunMat = new Matrix(3, 3);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				trueFunMat.set(i, j, Double.parseDouble(trueFunMatLine[i * 3 + j]));
			}
		}
		sample.trueFundamentalMatrix = trueFunMat;
		line++;

		// estimated fundamental matrix
		String[] estFunMatLine = lines[line].split(",");
		Matrix estFunMat = new Matrix(3, 3);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				estFunMat.set(i, j, Double.parseDouble(estFunMatLine[i * 3 + j]));
			}
		}
		sample.estimatedFundamentalMatrix = estFunMat;
		line++;

		// estimated homography matrix
		String[] estHomLine = lines[line].split(",");
		Mat estHom = new Mat(3, 3, CvType.CV_64F);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				estHom.put(i, j, Double.parseDouble(estHomLine[i * 3 + j]));
			}
		}
		sample.estimatedHomography = estHom;
		line++;

		// estimated essential matrix
		String[] estEssentialMatLine = lines[line].split(",");
		Matrix estEssentialMat = new Matrix(3, 3);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				estEssentialMat.set(i, j, Double.parseDouble(estEssentialMatLine[i * 3 + j]));
			}
		}
		sample.estimatedEssentialMatrix = estEssentialMat;
		line++;

		// poseTrueFun
		String[] poseTrueFunLine = lines[line].split(",");
		Matrix poseTrueFun = new Matrix(3, 4);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				poseTrueFun.set(i, j, Double.parseDouble(poseTrueFunLine[i * 3 + j]));
			}
		}
		sample.poseTrueFun = poseTrueFun;
		line++;

		// poseEstFun
		String[] poseEstFunLine = lines[line].split(",");
		Matrix poseEstFun = new Matrix(3, 4);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				poseEstFun.set(i, j, Double.parseDouble(poseEstFunLine[i * 3 + j]));
			}
		}
		sample.poseEstFun = poseEstFun;
		line++;

		// poseEstHomography
		String[] poseEstHomographyLine = lines[line].split(",");
		Matrix poseEstHomography = new Matrix(3, 4);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				poseEstHomography.set(i, j, Double.parseDouble(poseEstHomographyLine[i * 3 + j]));
			}
		}
		sample.poseEstHomography = poseEstHomography;
		line++;

		// poseEstEssential
		String[] poseEstEssentialLine = lines[line].split(",");
		Matrix poseEstEssential = new Matrix(3, 4);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				poseEstEssential.set(i, j, Double.parseDouble(poseEstEssentialLine[i * 3 + j]));
			}
		}
		sample.poseEstEssential = poseEstEssential;
		line++;

		// estPointsTrue
		String[] estPointsTrueLine = lines[line].split("\\|");
		List<Matrix> estPointsTrue = new ArrayList<Matrix>(estPointsTrueLine.length);
		for (int i = 0; i < estPointsTrueLine.length; i++) {
			String[] coords = estPointsTrueLine[i].split(",");
			Matrix p = new Matrix(4, 1);
			p.set(0, 0, Double.parseDouble(coords[0]));
			p.set(1, 0, Double.parseDouble(coords[1]));
			p.set(2, 0, Double.parseDouble(coords[2]));
			p.set(3, 0, 1);
			estPointsTrue.add(p);
		}
		sample.estPointsTrue = estPointsTrue;
		line++;

		// estPointsTrueFun
		String[] estPointsTrueFunLine = lines[line].split("\\|");
		List<Matrix> estPointsTrueFun = new ArrayList<Matrix>(estPointsTrueFunLine.length);
		for (int i = 0; i < estPointsTrueFunLine.length; i++) {
			String[] coords = estPointsTrueFunLine[i].split(",");
			Matrix p = new Matrix(4, 1);
			p.set(0, 0, Double.parseDouble(coords[0]));
			p.set(1, 0, Double.parseDouble(coords[1]));
			p.set(2, 0, Double.parseDouble(coords[2]));
			p.set(3, 0, 1);
			estPointsTrueFun.add(p);
		}
		sample.estPointsTrueFun = estPointsTrueFun;
		line++;

		// estPointsEstFun
		String[] estPointsEstFunLine = lines[line].split("\\|");
		List<Matrix> estPointsEstFun = new ArrayList<Matrix>(estPointsEstFunLine.length);
		for (int i = 0; i < estPointsEstFunLine.length; i++) {
			String[] coords = estPointsEstFunLine[i].split(",");
			Matrix p = new Matrix(4, 1);
			p.set(0, 0, Double.parseDouble(coords[0]));
			p.set(1, 0, Double.parseDouble(coords[1]));
			p.set(2, 0, Double.parseDouble(coords[2]));
			p.set(3, 0, 1);
			estPointsEstFun.add(p);
		}
		sample.estPointsEstFun = estPointsEstFun;
		line++;

		// estPointsEstHomography
		String[] estPointsEstHomographyLine = lines[line].split("\\|");
		List<Matrix> estPointsEstHomography = new ArrayList<Matrix>(estPointsEstHomographyLine.length);
		for (int i = 0; i < estPointsEstHomographyLine.length; i++) {
			String[] coords = estPointsEstHomographyLine[i].split(",");
			Matrix p = new Matrix(4, 1);
			p.set(0, 0, Double.parseDouble(coords[0]));
			p.set(1, 0, Double.parseDouble(coords[1]));
			p.set(2, 0, Double.parseDouble(coords[2]));
			p.set(3, 0, 1);
			estPointsEstHomography.add(p);
		}
		sample.estPointsEstHomography = estPointsEstHomography;
		line++;

		// estPointsEstEssential
		String[] estPointsEstEssentialLine = lines[line].split("\\|");
		List<Matrix> estPointsEstEssential = new ArrayList<Matrix>(estPointsEstEssentialLine.length);
		for (int i = 0; i < estPointsEstEssentialLine.length; i++) {
			String[] coords = estPointsEstEssentialLine[i].split(",");
			Matrix p = new Matrix(4, 1);
			p.set(0, 0, Double.parseDouble(coords[0]));
			p.set(1, 0, Double.parseDouble(coords[1]));
			p.set(2, 0, Double.parseDouble(coords[2]));
			p.set(3, 0, 1);
			estPointsEstEssential.add(p);
		}
		sample.estPointsEstEssential = estPointsEstEssential;
		line++;

		// reconstruction errors
		String[] reconstErrorLine = lines[line].split(",");
		sample.totalReconstErrorTrue = Double.parseDouble(reconstErrorLine[0]);
		sample.totalReconstErrorTrueFun = Double.parseDouble(reconstErrorLine[1]);
		sample.totalReconstErrorEstFun = Double.parseDouble(reconstErrorLine[2]);
		sample.totalReconstErrorEstHomography = Double.parseDouble(reconstErrorLine[3]);
		sample.totalReconstErrorEstEssential = Double.parseDouble(reconstErrorLine[4]);
		line++;

		reconstErrorLine = lines[line].split(",");
		sample.medianReconstErrorTrue = Double.parseDouble(reconstErrorLine[0]);
		sample.medianReconstErrorTrueFun = Double.parseDouble(reconstErrorLine[1]);
		sample.medianReconstErrorEstFun = Double.parseDouble(reconstErrorLine[2]);
		sample.medianReconstErrorEstHomography = Double.parseDouble(reconstErrorLine[3]);
		sample.medianReconstErrorEstEssential = Double.parseDouble(reconstErrorLine[4]);
		line++;

		// reprojection errors
		String[] reprojErrorLine = lines[line].split(",");
		sample.totalReprojErrorTrue = Double.parseDouble(reprojErrorLine[0]);
		sample.totalReprojErrorTrueFun = Double.parseDouble(reprojErrorLine[1]);
		sample.totalReprojErrorEstFun = Double.parseDouble(reprojErrorLine[2]);
		sample.totalReprojErrorEstHomography = Double.parseDouble(reprojErrorLine[3]);
		sample.totalReprojErrorEstEssential = Double.parseDouble(reprojErrorLine[4]);
		line++;

		// rotational chordal distances
		String[] rotChordalLine = lines[line].split(",");
		sample.rotChordalTrueFun = Double.parseDouble(rotChordalLine[0]);
		sample.rotChordalEstFun = Double.parseDouble(rotChordalLine[1]);
		sample.rotChordalEstHomography = Double.parseDouble(rotChordalLine[2]);
		sample.rotChordalEstEssential = Double.parseDouble(rotChordalLine[3]);
		line++;

		// translational chordal distances
		String[] transChordalLine = lines[line].split(",");
		sample.transChordalTrueFun = Double.parseDouble(transChordalLine[0]);
		sample.transChordalEstFun = Double.parseDouble(transChordalLine[1]);
		sample.transChordalEstHomography = Double.parseDouble(transChordalLine[2]);
		sample.transChordalEstEssential = Double.parseDouble(transChordalLine[3]);

		return sample;
	}

	public void bundleAdjust() {

		// estimated fundamental matrix
		Pose poseEstFunPose = Utils.matrixToPose(this.poseEstFun);
		BundleAdjustor.bundleAdjustPair(this.primaryCamera, poseEstFunPose, this.estPointsEstFun, this.correspondences,
				10);
		this.poseEstFun = poseEstFunPose.getHomogeneousMatrix();

		// estimated essential matrix
		Pose poseEstEssentialPose = Utils.matrixToPose(this.poseEstEssential);
		BundleAdjustor.bundleAdjustPair(this.primaryCamera, poseEstEssentialPose, this.estPointsEstEssential,
				this.correspondences, 10);
		this.poseEstEssential = poseEstEssentialPose.getHomogeneousMatrix();

		// estimated essential matrix
		Pose poseEstHomographyPose = Utils.matrixToPose(this.poseEstHomography);
		BundleAdjustor.bundleAdjustPair(this.primaryCamera, poseEstHomographyPose, this.estPointsEstHomography,
				this.correspondences, 10);
		this.poseEstHomography = poseEstHomographyPose.getHomogeneousMatrix();

		// re-evaluate metrics
		this.errorMetrics();

	}

	public void errorMetrics() {

		CameraParams cameraParams = new CameraParams();

		// reconstruction errors
		DoubleWrapper medianWrapper = new DoubleWrapper();
		this.totalReconstErrorTrue = ComputerVision.totalReconstructionError(this.estPointsTrue, this.truePoints,
				medianWrapper);
		this.medianReconstErrorTrue = medianWrapper.value;
		this.totalReconstErrorTrueFun = ComputerVision.totalReconstructionError(this.estPointsTrueFun, this.truePoints,
				medianWrapper);
		this.medianReconstErrorTrueFun = medianWrapper.value;
		this.totalReconstErrorEstFun = ComputerVision.totalReconstructionError(this.estPointsEstFun, this.truePoints,
				medianWrapper);
		this.medianReconstErrorEstFun = medianWrapper.value;
		this.totalReconstErrorEstHomography = ComputerVision.totalReconstructionError(this.estPointsEstHomography,
				this.truePoints, medianWrapper);
		this.medianReconstErrorEstHomography = medianWrapper.value;
		this.totalReconstErrorEstEssential = ComputerVision.totalReconstructionError(this.estPointsEstEssential,
				this.truePoints, medianWrapper);
		this.medianReconstErrorEstEssential = medianWrapper.value;

		// reprojection errors
		this.totalReprojErrorTrue = ComputerVision.getTotalReprojectionError(
				this.secondaryCamera.getHomogeneousMatrix(), this.primaryCamera.getHomogeneousMatrix(), cameraParams,
				correspondences, this.estPointsTrue);
		this.totalReprojErrorTrueFun = ComputerVision.getTotalReprojectionError(this.poseTrueFun,
				this.primaryCamera.getHomogeneousMatrix(), cameraParams, correspondences, this.estPointsTrueFun);
		this.totalReprojErrorEstFun = ComputerVision.getTotalReprojectionError(this.poseEstFun,
				this.primaryCamera.getHomogeneousMatrix(), cameraParams, correspondences, this.estPointsEstFun);
		this.totalReprojErrorEstHomography = ComputerVision.getTotalReprojectionError(this.poseEstHomography,
				this.primaryCamera.getHomogeneousMatrix(), cameraParams, correspondences, this.estPointsEstHomography);
		this.totalReprojErrorEstEssential = ComputerVision.getTotalReprojectionError(this.poseEstEssential,
				this.primaryCamera.getHomogeneousMatrix(), cameraParams, correspondences, this.estPointsEstEssential);

		// chordal distances
		this.rotChordalTrueFun = Utils.chordalDistance(poseTrueFun.getMatrix(0, 2, 0, 2),
				this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 0, 2));
		this.rotChordalEstFun = Utils.chordalDistance(poseEstFun.getMatrix(0, 2, 0, 2),
				this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 0, 2));
		this.rotChordalEstHomography = Utils.chordalDistance(poseEstHomography.getMatrix(0, 2, 0, 2),
				this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 0, 2));
		this.rotChordalEstEssential = Utils.chordalDistance(poseEstEssential.getMatrix(0, 2, 0, 2),
				this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 0, 2));

		this.transChordalTrueFun = Utils.chordalDistance(poseTrueFun.getMatrix(0, 2, 3, 3).times(
				1 / (poseTrueFun.getMatrix(0, 2, 3, 3).normF() > 0 ? poseTrueFun.getMatrix(0, 2, 3, 3).normF() : 1)),
				this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 3, 3)
						.times(1 / (this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 3, 3).normF() > 0
								? this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 3, 3).normF()
								: 1)));

		this.transChordalEstFun = Utils.chordalDistance(poseEstFun.getMatrix(0, 2, 3, 3).times(
				1 / (poseEstFun.getMatrix(0, 2, 3, 3).normF() > 0 ? poseEstFun.getMatrix(0, 2, 3, 3).normF() : 1)),
				this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 3, 3)
						.times(1 / (this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 3, 3).normF() > 0
								? this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 3, 3).normF()
								: 1)));
		this.transChordalEstHomography = Utils.chordalDistance(
				poseEstHomography.getMatrix(0, 2, 3, 3)
						.times(1 / (poseEstHomography.getMatrix(0, 2, 3, 3).normF() > 0
								? poseEstHomography.getMatrix(0, 2, 3, 3).normF()
								: 1)),
				this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 3, 3)
						.times(1 / (this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 3, 3).normF() > 0
								? this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 3, 3).normF()
								: 1)));
		this.transChordalEstEssential = Utils.chordalDistance(
				poseEstEssential.getMatrix(0, 2, 3, 3)
						.times(1 / (poseEstEssential.getMatrix(0, 2, 3, 3).normF() > 0
								? poseEstEssential.getMatrix(0, 2, 3, 3).normF()
								: 1)),
				this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 3, 3)
						.times(1 / (this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 3, 3).normF() > 0
								? this.secondaryCamera.getHomogeneousMatrix().getMatrix(0, 2, 3, 3).normF()
								: 1)));

	}

}
