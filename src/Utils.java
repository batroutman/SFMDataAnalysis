import org.opencv.core.CvType;
import org.opencv.core.Mat;

import Jama.Matrix;

public class Utils {

	public static void pl(Object obj) {
		System.out.println(obj);
	}

	public static void p(Object obj) {
		System.out.print(obj);
	}

	// quaternion multiplication, assuming column vector of format [qw, qx, qy,
	// qz].transpose() (q1*q2)
	public static Matrix quatMult(Matrix q1, Matrix q2) {

		Matrix t = new Matrix(4, 1);

		double q1w = q1.get(0, 0);
		double q1x = q1.get(1, 0);
		double q1y = q1.get(2, 0);
		double q1z = q1.get(3, 0);

		double q2w = q2.get(0, 0);
		double q2x = q2.get(1, 0);
		double q2y = q2.get(2, 0);
		double q2z = q2.get(3, 0);

		t.set(1, 0, q1x * q2w + q1y * q2z - q1z * q2y + q1w * q2x);
		t.set(2, 0, -q1x * q2z + q1y * q2w + q1z * q2x + q1w * q2y);
		t.set(3, 0, q1x * q2y - q1y * q2x + q1z * q2w + q1w * q2z);
		t.set(0, 0, -q1x * q2x - q1y * q2y - q1z * q2z + q1w * q2w);

		t = t.times(1 / t.normF());

		return t;
	}

	public static Matrix MatToMatrix(Mat mat) {
		if (mat.type() == 6) {
			double[] buffer = new double[mat.rows() * mat.cols()];
			mat.get(0, 0, buffer);
			Matrix matrix = new Matrix(mat.rows(), mat.cols());
			for (int i = 0; i < matrix.getRowDimension(); i++) {
				for (int j = 0; j < matrix.getColumnDimension(); j++) {
					matrix.set(i, j, buffer[i * mat.cols() + j]);
				}
			}

			return matrix;
		} else {
			// 0
			byte[] buffer = new byte[mat.rows() * mat.cols()];
			mat.get(0, 0, buffer);
			Matrix matrix = new Matrix(mat.rows(), mat.cols());
			for (int i = 0; i < matrix.getRowDimension(); i++) {
				for (int j = 0; j < matrix.getColumnDimension(); j++) {
					matrix.set(i, j, Byte.toUnsignedInt(buffer[i * mat.cols() + j]));
				}
			}

			return matrix;
		}

	}

	public static Mat MatrixToMat(Matrix matrix) {
		Mat mat = new Mat(matrix.getRowDimension(), matrix.getColumnDimension(), CvType.CV_64F);
		for (int row = 0; row < matrix.getRowDimension(); row++) {
			for (int col = 0; col < matrix.getColumnDimension(); col++) {
				mat.put(row, col, matrix.get(row, col));
			}
		}
		return mat;
	}

	public static void printMatrix(Mat mat) {
		double[] buffer = new double[mat.rows() * mat.cols()];
		mat.get(0, 0, buffer);
		for (int i = 0; i < mat.rows(); i++) {
			for (int j = 0; j < mat.cols(); j++) {
				p(buffer[mat.cols() * i + j] + (j == mat.cols() - 1 ? "" : " ,"));
			}
			pl("");
		}
	}

	public static double chordalDistance(Matrix R1, Matrix R2) {
		return R1.minus(R2).normF();
	}

	public static double quaternionDistance(Matrix q1, Matrix q2) {

		return Math.min(q1.plus(q2).normF(), q1.minus(q2).normF());

	}

	// return a Pose P such that P * pose0 = pose1
	public static Pose getPoseDifference(Pose pose0, Pose pose1) {
		Matrix q1 = new Matrix(4, 1);
		q1.set(0, 0, pose1.getQw());
		q1.set(1, 0, pose1.getQx());
		q1.set(2, 0, pose1.getQy());
		q1.set(3, 0, pose1.getQz());

		// invert the initial pose quaternion
		Matrix q0Inv = new Matrix(4, 1);
		q0Inv.set(0, 0, -pose0.getQw());
		q0Inv.set(1, 0, pose0.getQx());
		q0Inv.set(2, 0, pose0.getQy());
		q0Inv.set(3, 0, pose0.getQz());

		// calculate the new rotation difference from pose0 -> pose1
		Matrix newQuat = quatMult(q1, q0Inv);
//		pl("q1: " + q1.get(0, 0) + ", " + q1.get(1, 0) + ", " + q1.get(2, 0) + ", " + q1.get(3, 0));
//		pl("q0Inv: " + q0Inv.get(0, 0) + ", " + q0Inv.get(1, 0) + ", " + q0Inv.get(2, 0) + ", " + q0Inv.get(3, 0));
//		pl("quatMult: " + newQuat.get(0, 0) + ", " + newQuat.get(1, 0) + ", " + newQuat.get(2, 0) + ", "
//				+ newQuat.get(3, 0));

		// calculate absolute translation difference
		double cx = pose1.getCx() - pose0.getCx();
		double cy = pose1.getCy() - pose0.getCy();
		double cz = pose1.getCz() - pose0.getCz();

		Pose newPose = new Pose();
		newPose.setQw(newQuat.get(0, 0));
		newPose.setQx(newQuat.get(1, 0));
		newPose.setQy(newQuat.get(2, 0));
		newPose.setQz(newQuat.get(3, 0));
		newPose.setCx(cx);
		newPose.setCy(cy);
		newPose.setCz(cz);
		return newPose;
	}

}
