package net.rem.regression;

import java.util.List;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import net.hudup.core.Util;
import net.hudup.core.logistic.DSUtil;
import net.hudup.core.logistic.LogUtil;

/**
 * This is utility class provides mathematical functions.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class MathAdapter {

	
	/**
	 * Calculating determinant of the given matrix.
	 * @param A given matrix.
	 * @return determinant of the given matrix.
	 */
	public static double matrixDeterminant(List<double[]> A) {
		RealMatrix M = MatrixUtils.createRealMatrix(A.toArray(new double[A.size()][A.size()]));
		LUDecomposition lu = new LUDecomposition(M);
		return lu.getDeterminant();
	}
	
	
	/**
	 * Calculating inverse of the given matrix.
	 * @param A given matrix.
	 * @return inverse of the given matrix. Return null if the matrix is not invertible.
	 */
	public static List<double[]> matrixInverse(List<double[]> A) {
		RealMatrix M = MatrixUtils.createRealMatrix(A.toArray(new double[A.size()][A.size()]));
		try {
			//Firstly, solve exact solution by LU Decomposition
			DecompositionSolver solver = new LUDecomposition(M).getSolver();
			return DSUtil.toDoubleList(solver.getInverse().getData());
		}
		catch (Exception e1) {
			LogUtil.info("Problem from LU Decomposition: " + e1.getMessage());
			try {
				//Secondly, solve approximate solution by QR Decomposition
				DecompositionSolver solver = new QRDecomposition(M).getSolver(); //It is possible to replace QRDecomposition by LUDecomposition here.
				return DSUtil.toDoubleList(solver.getInverse().getData());
			}
			catch (SingularMatrixException e2) {
				LogUtil.info("Singular matrix problem from QR Decomposition");
				//Finally, solve approximate solution by Moore–Penrose pseudo-inverse matrix
				try {
					DecompositionSolver solver = new SingularValueDecomposition(M).getSolver(); //It is possible to replace QRDecomposition by LUDecomposition here.
					return DSUtil.toDoubleList(solver.getInverse().getData());
				}
				catch (SingularMatrixException e3) {
					LogUtil.info("Cannot solve the problem of singluar matrix by Moore–Penrose pseudo-inverse matrix in #solve(RealMatrix, RealVector)");
				}
			}
		}
		
		return Util.newList();
	}

	
//	/**
//	 * Checking if the given matrix is invertible.
//	 * @param A given matrix.
//	 * @return if the given matrix is invertible.
//	 */
//	public static boolean matrixIsInvertible(List<double[]> A) {
//		RealMatrix M = MatrixUtils.createRealMatrix(A.toArray(new double[A.size()][A.size()]));
//		DecompositionSolver solver = new LUDecomposition(M).getSolver();
//		return solver.isNonSingular();
//	}
	
	
	/**
	 * Solving the equation Ax = b. This method uses firstly LU decomposition to solve exact solution and then uses QR decomposition to solve approximate solution in least square sense.
	 * Finally, if the problem of singular matrix continues to raise, Moore–Penrose pseudo-inverse matrix is used to find approximate solution.
	 * @param A specified matrix.
	 * @param b specified vector.
	 * @return solution x of the equation Ax = b. Return null if any error raises.
	 */
	public static List<Double> solve(List<double[]> A, List<Double> b) {
		int N = b.size();
		int n = A.get(0).length;
		if (N == 0 || n == 0)
			return null;
		
		List<Double> x = null;
		RealMatrix M = MatrixUtils.createRealMatrix(A.toArray(new double[N][n]));
		RealVector m = new ArrayRealVector(b.toArray(new Double[] {}));
		try {
			//Firstly, solve exact solution by LU Decomposition
			DecompositionSolver solver = new LUDecomposition(M).getSolver();
			x = DSUtil.toDoubleList(solver.solve(m).toArray()); //solve Ax = b exactly
			x = checkSolution(x);
			if (x == null)
				throw new Exception("Null solution");
		}
		catch (Exception e1) {
			LogUtil.info("Problem from LU Decomposition: " + e1.getMessage());
			try {
				//Secondly, solve approximate solution by QR Decomposition
				DecompositionSolver solver = new QRDecomposition(M).getSolver(); //It is possible to replace QRDecomposition by LUDecomposition here.
				x = DSUtil.toDoubleList(solver.solve(m).toArray()); //solve Ax = b with approximation
				x = checkSolution(x);
			}
			catch (SingularMatrixException e2) {
				LogUtil.info("Singular matrix problem from QR Decomposition");
				//Finally, solve approximate solution by Moore–Penrose pseudo-inverse matrix
				try {
					DecompositionSolver solver = new SingularValueDecomposition(M).getSolver(); //It is possible to replace QRDecomposition by LUDecomposition here.
					RealMatrix pseudoInverse = solver.getInverse();
					x = DSUtil.toDoubleList(pseudoInverse.operate(m).toArray());
					x = checkSolution(x);
				}
				catch (SingularMatrixException e3) {
					LogUtil.info("Cannot solve the problem of singluar matrix by Moore–Penrose pseudo-inverse matrix in #solve(RealMatrix, RealVector)");
					x = null;
				}
			}
		}
		
		return x;
	}


	/**
	 * Checking the specified solution of equation Ax = b.
	 * @param x the specified solution.
	 * @return the checked solution.
	 */
	private static List<Double> checkSolution(List<Double> x) {
		if (x == null)
			return null;
		for (int i = 0; i < x.size(); i++) {
			Double value = x.get(i);
			if (value == null || Double.isNaN(value) || !Util.isUsed(value))
				return null;
		}
		return x;
	}
	
	
	/**
	 * Evaluating the normal probability density function with specified mean and variance.
	 * Inherited class can re-defined this density function.
	 * @param value specified response value z.
	 * @param mean specified mean.
	 * @param variance specified variance.
	 * @return value evaluated from the normal probability density function.
	 */
	public static double normalPDF(double value, double mean, double variance) {
		if (variance == 0 && mean != value) return 0;
		if (variance == 0 && mean == value) return 1;
		
//		variance = variance != 0 ? variance : Float.MIN_VALUE;
		double d = value - mean;
		return (1.0 / (Math.sqrt(2*Math.PI*variance))) * Math.exp(-(d*d) / (2*variance));
	}

	
	/**
	 * Evaluating the logarithm normal probability density function with specified mean and variance.
	 * Inherited class can re-defined this density function.
	 * @param value specified response value z.
	 * @param mean specified mean.
	 * @param variance specified variance.
	 * @return value evaluated from the logarithm normal probability density function.
	 */
	public static double logNormalPDF(double value, double mean, double variance) {
		if (variance == 0 && mean != value) return Double.NEGATIVE_INFINITY;
		if (variance == 0 && mean == value) return 0;
		
//		variance = variance != 0 ? variance : Float.MIN_VALUE;
		double d = value - mean;
        double logVarPi = Math.log(2*Math.PI) + Math.log(variance);
        return -0.5*(logVarPi + d*d/variance);

	}

	
	/**
	 * Evaluating the normal cumulative density function with specified mean and variance.
	 * Inherited class can re-defined this density function.
	 * @param value specified response value z.
	 * @param mean specified mean.
	 * @param variance specified variance.
	 * @return value evaluated from the normal probability density function.
	 */
	public static double normalCDF(double value, double mean, double variance) {
		if (variance == 0 && mean != value) return 0;
		if (variance == 0 && mean == value) return 1;
		
//		variance = variance != 0 ? variance : Float.MIN_VALUE;
		return new NormalDistribution(mean, Math.sqrt(variance)).cumulativeProbability(value);
	}

	
	/**
	 * Evaluating the logarithm normal cumulative density function with specified mean and variance.
	 * Inherited class can re-defined this density function.
	 * @param value specified response value z.
	 * @param mean specified mean.
	 * @param variance specified variance.
	 * @return value evaluated from the logarithm normal probability density function.
	 */
	public static double logNormalCDF(double value, double mean, double variance) {
		double cdf = normalCDF(value, mean, variance);
		if (cdf == 0)
			return Double.NEGATIVE_INFINITY;
		else
			return Math.log(cdf);
	}

	
	/**
	 * Main method.
	 * @param args array of arguments.
	 */
	public static void main(String[] args) {
		double a = Math.log(normalPDF(1, 2, 3));
		double b = logNormalPDF(1, 2, 3);
		System.out.println("a=" + a + ", b=" + b);
	}
	
	
}
