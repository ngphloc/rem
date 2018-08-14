package net.hudup.regression;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import com.speqmath.Parser;

import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.AbstractTestingAlg;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.MathUtil;
import net.hudup.core.parser.TextParserUtil;

/**
 * <code>AbstractRegression</code> is the most abstract class for expectation maximization (EM) algorithm.
 * It implements partially the interface {@link Regression}.
 * 
 * @author Loc Nguyen
 * @version 1.0*
 */
public abstract class AbstractRegression extends AbstractTestingAlg implements Regression {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Name of regression indices field.
	 */
	public final static String R_INDICES_FIELD = "r_indices";

	
	/**
	 * Default regression indices field.
	 */
	public final static String R_INDICES_FIELD_DEFAULT = "{-1+, -1+}, {-1+}, {-1}"; //Use default indices in which n-1 first variables are regressors and the last variable is response variable
	
	
	/**
	 * Regression coefficient.
	 */
	protected double[] coeffs = null;
	
	
	/**
	 * Indices for X data.
	 */
	protected List<Object[]> xIndices = new ArrayList<>();

	
	/**
	 * Indices for Z data.
	 */
	protected List<Object[]> zIndices = new ArrayList<>();
	
	
	/**
	 * Attribute list for all variables: all X, Y, and z.
	 */
	protected AttributeList attList = null;
	

    /**
     * Default constructor
     */
	public AbstractRegression() {
		super();
		// TODO Auto-generated constructor stub
	}


	@Override
	public synchronized Object execute(Object input) {
		// TODO Auto-generated method stub
		if (this.coeffs == null)
			return null;
		
		if (input == null || !(input instanceof Profile))
			return null; //only support profile input currently
		Profile profile = (Profile)input;
		
		double sum = this.coeffs[0];
		for (int j= 0; j < this.coeffs.length - 1; j++) {
			double value = extractRegressor(profile, j + 1); //due to x = (1, x1, x2,..., xn) and xIndices.get(0) = -1
			sum += this.coeffs[j + 1] * (double)transformRegressor(value, false); 
		}
		
		return transformResponse(sum, true);
	}

	
	@Override
	public synchronized Object getParameter() {
		// TODO Auto-generated method stub
		return coeffs;
	}

	
	@Override
	public DataConfig createDefaultConfig() {
		// TODO Auto-generated method stub
		DataConfig config = super.createDefaultConfig();
		config.put(R_INDICES_FIELD, R_INDICES_FIELD_DEFAULT); //Not used
		return config;
	}


	@Override
	public String parameterToShownText(Object parameter, Object... info) {
		// TODO Auto-generated method stub
		if (parameter == null || !(parameter instanceof double[]))
			return "";
		double[] coeffs = (double[])parameter;

		StringBuffer buffer = new StringBuffer();
		for (int j = 0; j < coeffs.length; j++) {
			if (j > 0)
				buffer.append(", ");
			buffer.append(MathUtil.format(coeffs[j]));
		}
		
		return buffer.toString();
	}

	
	@Override
	public synchronized String getDescription() {
		// TODO Auto-generated method stub
		if (this.coeffs == null)
			return "";
		
		StringBuffer buffer = new StringBuffer();
		buffer.append(transformResponse(extractResponseName(), false) + " = " + MathUtil.format(coeffs[0]));
		for (int j = 0; j < this.coeffs.length - 1; j++) {
			double coeff = this.coeffs[j + 1];
			String variableName = transformRegressor(extractRegressorName(j + 1), false).toString();
			if (coeff < 0)
				buffer.append(" - " + MathUtil.format(Math.abs(coeff)) + "*" + variableName);
			else
				buffer.append(" + " + MathUtil.format(coeff) + "*" + variableName);
		}
		
		return buffer.toString();
	}
	
	
	/**
	 * Extracting value of regressor (X) from specified profile.
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @param profile specified profile.
	 * @param index specified indices.
	 * @return value of regressor (X) extracted from specified profile.
	 */
	protected double extractRegressor(Profile profile, int index) {
		// TODO Auto-generated method stub
		return defaultExtractVariable(profile, xIndices, index);
	}


	/**
	 * Extracting name of regressor (X).
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @param index specified indices.
	 * @return text of regressor (X) extracted.
	 */
	protected String extractRegressorName(int index) {
		// TODO Auto-generated method stub
		return defaultExtractVariableName(attList, xIndices, index);
	}


	/**
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 */
	@Override
	public double extractResponse(Profile profile) {
		// TODO Auto-generated method stub
		return defaultExtractVariable(profile, zIndices, 1);
	}


	/**
	 * Extracting name of response variable (Z).
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @return text of response variable (Z) extracted.
	 */
	protected String extractResponseName() {
		// TODO Auto-generated method stub
		return defaultExtractVariableName(attList, zIndices, 1);
	}


	/**
	 * Transforming independent variable X.
	 * In the most general case that each index is an mathematical expression, this method is not focused.
	 * @param x specified variable X.
	 * @param inverse if true, there is an inverse transformation.
	 * @return transformed value of X.
	 */
	protected Object transformRegressor(Object x, boolean inverse) {
		// TODO Auto-generated method stub
		return x;
	}


	/**
	 * Transforming independent variable Z.
	 * In the most general case that each index is an mathematical expression, this method is not focused.
	 * @param z specified variable Z.
	 * @param inverse if true, there is an inverse transformation.
	 * @return transformed value of Z.
	 */
	protected Object transformResponse(Object z, boolean inverse) {
		// TODO Auto-generated method stub
		return z;
	}


	/**
	 * Standardizing attribute names, for example, if attribute name is an number name "1.0", it is converted into "a1.0".
	 * @param attList standardized attribute list.
	 */
	public static void standardizeAttributeNames(AttributeList attList) {
//		for (int i = 0; i < attList.size(); i++) {
//			String name = attList.get(i).getName();
//			try {
//				Double.parseDouble(name);
//				System.out.println("Attribute name \"" + name + "\" is invalid because it is a number");
//				attList.get(i).setName("a" + name);
//			}
//			catch (Throwable e) {
//				
//			}
//		}
	}
	
	
	/**
	 * Parsing indices of variables. In the most general case, each index is an mathematical expression.
	 * @return true if parsing is successful.
	 */
	public static boolean parseIndices(String cfgIndices, int maxVariables, List<Object[]> xIndices, List<Object[]> zIndices) {
		xIndices.clear();
		xIndices.add(new Object[] {new Integer(-1)}); // due to X = (1, x1, x2,..., x(n-1)) and there is no 1 in real data.
		zIndices.clear();
		zIndices.add(new Object[] {new Integer(-1)}); // due to Z = (1, z) and there is no 1 in real data.
		
		//Begin extracting indices from configuration.
		//The pattern is {1, 2}, {3, 4, 5), {5, 6}, {5, 6, 7, 8}, {9, 10}
		//The pattern can also be 1, 2, 3, 4, 5, 5, 6, 5, 6, 7, 8, 9, 10
		List<Object[]> indices = new ArrayList<>();
		if (cfgIndices != null && !cfgIndices.isEmpty() && !cfgIndices.equals(R_INDICES_FIELD_DEFAULT)) {
			String regex = "\\}(\\s)*,(\\s)*\\{";
			//String regex = "\\)(\\s)*,(\\s)*\\(";
			String[] txtArray = cfgIndices.split(regex);
			List<String> txtList = new ArrayList<>();
			for (String txt : txtArray) {
				txt = txt.trim().replaceAll("\\}", "").replaceAll("\\{", "").replaceAll("\\)", "").replaceAll("\\(", "");
				if (!txt.isEmpty())
					txtList.add(txt);
			}
			if (txtList.size() == 1) { //The case: 1, 2, 3, 4, 5, 5, 6, 5, 6, 7, 8, 9, 10
				List<Object> oneIndices = parseIndex(txtList.get(0), ",");
				for (Object index : oneIndices)
					indices.add(new Object[] {index});
			}
			else if (txtList.size() > 1) {
				for (String txt : txtList) {
					List<Object> oneIndices = parseIndex(txt, ",");
					if (oneIndices.size() == 0)
						continue;
					indices.add(oneIndices.toArray());
				}
			}
		}
		
		if (indices == null || indices.size() < 2) {
			for (int j = 0; j < maxVariables - 1; j++)
				xIndices.add(new Object[] {new Integer(j)});
			zIndices.add(new Object[] {new Integer(maxVariables - 1)}); //The last index is Z index.
		}
		else {
			for (int j = 0; j < indices.size() - 1; j++)
				xIndices.add(indices.get(j));
			zIndices.add(indices.get(indices.size() - 1)); //The last index is Z index
		}
		//End extracting indices from configuration
		
		if (zIndices.size() < 2 || xIndices.size() < 2)
			return false;
		else
			return true;
	}

	
	/**
	 * Parsing each index.
	 * @param txtIndex text of index.
	 * @param sep separated string.
	 * @return list of indices parsed from text.
	 */
	private static List<Object> parseIndex(String txtIndex, String sep) {
		List<Object> indices = new ArrayList<>();
		if (txtIndex == null || txtIndex.isEmpty())
			return indices;
		List<String> array = TextParserUtil.split(txtIndex, sep, null);
		
		for (String el : array) {
			int index = -1;
			boolean parseSuccess = true;
			try {
				index = Integer.parseInt(el);
			}
			catch (Throwable e) {
				parseSuccess = false;
			}
			if (parseSuccess)
				indices.add(new Integer(index));
			else
				indices.add(el);
		}
		
		return indices;
	}

	
	/**
	 * Extracting value of variable (X) from specified profile.
	 * @param profile specified profile.
	 * @param indices specified list of indices.
	 * @param index specified index.
	 * @return value of variable (X) extracted from specified profile.
	 */
	public static double defaultExtractVariable(Profile profile, List<Object[]> indices, int index) {
		try {
			Object item = indices.get(index)[0];
			if (item instanceof Integer)
				return profile.getValueAsReal((int)item);
			
			String txtValue = item.toString().trim();
			int n = profile.getAttCount();
			for (int i = 0; i < n; i++) {
				String attName =  profile.getAtt(i).getName();
				if(!txtValue.contains(attName))
					continue;
				
				if(profile.isMissing(i))
					return Constants.UNUSED; //Cannot evaluate
				Double value = profile.getValueAsReal(attName);
				if(!Util.isUsed(value))
					return Constants.UNUSED; //Cannot evaluate
				
				txtValue = txtValue.replaceAll(attName, value.toString()).trim();
			}
			
			Parser parser = new Parser();
			return parser.parse2(txtValue);
		}
		catch (Throwable e) {
			e.printStackTrace();
		}
		
		return Constants.UNUSED;
	}
	
	
	/**
	 * Extracting variable name.
	 * @param attList specified attribute list.
	 * @param indices specified list of indices.
	 * @param index specified index.
	 * @return variable name.
	 */
	public static String defaultExtractVariableName(AttributeList attList, List<Object[]> indices, int index) {
		// TODO Auto-generated method stub
		Object item = indices.get(index)[0];
		if (item instanceof Integer)
			return attList.get((int)item).getName();
		else
			return item.toString();
	}
	

	/**
	 * Solving the equation Ax = b. This method uses firstly LU decomposition to solve exact solution and then uses QR decomposition to solve approximate solution in least square sense.
	 * Finally, if the problem of singular matrix continues to raise, Moore–Penrose pseudo-inverse matrix is used to find approximate solution.
	 * @param A specified matrix.
	 * @param b specified vector.
	 * @return solution x of the equation Ax = b. Return null if any error raises.
	 */
	public static double[] solve(List<double[]> A, double[] b) {
		int N = b.length;
		int n = A.get(0).length;
		if (N == 0 || n == 0)
			return null;
		
		double[] x = null;
		RealMatrix M = MatrixUtils.createRealMatrix(A.toArray(new double[N][n]));
		RealVector m = new ArrayRealVector(b);
		try {
			//Firstly, solve exact solution by LU Decomposition
			DecompositionSolver solver = new LUDecomposition(M).getSolver();
			x = solver.solve(m).toArray(); //solve Ax = b exactly
			x = checkSolution(x);
			if (x == null)
				throw new Exception("Null solution");
		}
		catch (Exception e1) {
			logger.info("Problem by LU Decomposition: " + e1.getMessage());
			try {
				//Secondly, solve approximate solution by QR Decomposition
				DecompositionSolver solver = new QRDecomposition(M).getSolver(); //It is possible to replace QRDecomposition by LUDecomposition here.
				x = solver.solve(m).toArray(); //solve Ax = b with approximation
				x = checkSolution(x);
			}
			catch (SingularMatrixException e2) {
				logger.info("Singular matrix problem by QR Decomposition");
				//Finally, solve approximate solution by Moore–Penrose pseudo-inverse matrix
				try {
					DecompositionSolver solver = new SingularValueDecomposition(M).getSolver(); //It is possible to replace QRDecomposition by LUDecomposition here.
					RealMatrix pseudoInverse = solver.getInverse();
					x = pseudoInverse.operate(m).toArray();
					x = checkSolution(x);
				}
				catch (SingularMatrixException e3) {
					logger.info("Cannot solve the problem of singluar matrix by Moore–Penrose pseudo-inverse matrix in #solve(RealMatrix, RealVector)");
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
	private static double[] checkSolution(double[] x) {
		if (x == null)
			return null;
		for (int i = 0; i < x.length; i++) {
			if (Double.isNaN(x[i]) || !Util.isUsed(x[i]))
				return null;
		}
		return x;
	}
	
	
}
