import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Mat;

import Jama.Matrix;

public class Sample {

	///////////////////////////////////////////////////////////////////////////////
	/////////////////////////// GIVEN INFORMATION /////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////

	public Pose primaryCamera = null;
	public Pose secondaryCamera = null;
	public List<Correspondence2D2D> correspondences = new ArrayList<Correspondence2D2D>();
	public Matrix trueFundamentalMatrix = null;

	///////////////////////////////////////////////////////////////////////////////
	/////////////////////////// ITEMS TO ESTIMATE /////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////

	public Matrix estimatedFundamentalMatrix = null;
	public Mat estimatedhomography = null;

	///////////////////////////////////////////////////////////////////////////////
	/////////////////////// SECONDARY POSES TO ESTIMATE ///////////////////////////
	///////////////////////////////////////////////////////////////////////////////

	// pose derived from true fundamental matrix
	public Matrix poseTrueFun = null;

	// pose derived from estimated fundamental matrix
	public Matrix poseEstFun = null;

	// pose derived from homography
	public Matrix poseEstHomography = null;

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

	///////////////////////////////////////////////////////////////////////////////
	/////////////////////////// CHORDAL DISTANCES /////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////

	// ---- rotational chordal distances ----
	public double rotChordalTrueFun = 0;
	public double rotChordalEstFun = 0;
	public double rotChordalEstHomography = 0;

	// -- normalized translational chordal distances ----
	public double transChordalTrueFun = 0;
	public double transChordalEstFun = 0;
	public double transChordalEstHomography = 0;

	public Sample() {
	}

	public void evaluate(VirtualEnvironment mock) {
		// ...
	}

}
