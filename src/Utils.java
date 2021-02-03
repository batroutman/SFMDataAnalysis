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

}
