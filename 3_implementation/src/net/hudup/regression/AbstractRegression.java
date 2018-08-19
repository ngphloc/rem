package net.hudup.regression;

import java.util.Arrays;
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
import net.hudup.core.data.Attribute;
import net.hudup.core.data.Attribute.Type;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.DSUtil;
import net.hudup.core.logistic.MathUtil;
import net.hudup.core.parser.TextParserUtil;

/**
 * This is the most abstract class for regression model. It implements partially the interface {@link Regression}.
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
	 * Regression coefficient.
	 */
	protected List<Double> coeffs = null;
	
	
	/**
	 * Indices for X data.
	 */
	protected List<Object[]> xIndices = Util.newList();

	
	/**
	 * Indices for Z data.
	 */
	protected List<Object[]> zIndices = Util.newList();
	
	
	/**
	 * Attribute list for all variables: all X, Y, and z.
	 * This variable is also used as the indicator of successful learning.
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
	public Object learn(Object...info) throws Exception {
		// TODO Auto-generated method stub
		Object resulted = null;
		if (prepareInternalData())
			resulted = learn0();
		if (resulted == null)
			clearInternalData();
		
		return resulted;
	}

	
	/**
	 * Internal learning parameters. Derived class needs to implement this method.
	 * @return the parameter to be learned.
	 * @throws Exception if any error occurs.
	 */
	protected abstract Object learn0() throws Exception;
	
	
	/**
	 * Preparing data.
	 * @return true if data preparation is successful.
	 * @throws Exception if any error raises.
	 */
	protected boolean prepareInternalData() throws Exception {
		// TODO Auto-generated method stub
		clearInternalData();
		
		Profile profile0 = null;
		if (this.sample.next()) {
			profile0 = this.sample.pick();
		}
		this.sample.reset();
		if (profile0 == null)
			return false;
		if (profile0.getAttCount() < 2) //x1, x2,..., x(n-1), z
			return false;
		this.attList = profile0.getAttRef();
		
		//Begin parsing indices
		String cfgIndices = this.getConfig().getAsString(R_INDICES_FIELD);
		if (!AbstractRegression.parseIndices(cfgIndices, profile0.getAttCount(), this.xIndices, this.zIndices))
			return false;
		//End parsing indices

		//Begin checking existence of values.
		boolean zExists = false;
		boolean[] xExists = new boolean[xIndices.size() - 1]; //profile = (x1, x2,..., x(n-1), z)
		Arrays.fill(xExists, false);
		while (this.sample.next()) {
			Profile profile = this.sample.pick(); //profile = (x1, x2,..., x(n-1), z)
			if (profile == null)
				continue;
			
			double lastValue = (double)extractResponse(profile);
			if (Util.isUsed(lastValue))
				zExists = zExists || true; 
			
			for (int j = 1; j < xIndices.size(); j++) {
				double value = extractRegressor(profile, j);
				if (Util.isUsed(value))
					xExists[j - 1] = xExists[j - 1] || true;
			}
		}
		this.sample.reset();

		List<Object[]> xIndicesTemp = Util.newList();
		xIndicesTemp.add(xIndices.get(0)); //adding -1
		for (int j = 1; j < xIndices.size(); j++) {
			if (xExists[j - 1])
				xIndicesTemp.add(xIndices.get(j)); //only use variables having at least one value.
		}
		if (!zExists || xIndicesTemp.size() < 2)
			return false;
		xIndices = xIndicesTemp;
		//End checking existence of values.

		return true;
	}

	
	/**
	 * Clear all internal data.
	 */
	protected void clearInternalData() {
		this.coeffs = null;
		this.xIndices.clear();
		this.zIndices.clear();
		this.attList = null;
	}

	
	@Override
	public synchronized Object execute(Object input) {
		// TODO Auto-generated method stub
		if (this.coeffs == null)
			return null;
		
		if (input == null || !(input instanceof Profile))
			return null; //only support profile input currently
		Profile profile = (Profile)input;
		
		double sum = this.coeffs.get(0);
		for (int j= 0; j < this.coeffs.size() - 1; j++) {
			double value = extractRegressor(profile, j + 1); //due to x = (1, x1, x2,..., xn) and xIndices.get(0) = -1
			sum += this.coeffs.get(j + 1) * (double)transformRegressor(value, false); 
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
		config.put(R_INDICES_FIELD, R_INDICES_FIELD_DEFAULT);
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
		buffer.append(transformResponse(extractResponseName(), false) + " = " + MathUtil.format(coeffs.get(0)));
		for (int j = 0; j < this.coeffs.size() - 1; j++) {
			double coeff = this.coeffs.get(j + 1);
			String regressorExpr = "(" + transformRegressor(extractRegressorName(j + 1), false).toString() + ")";
			if (coeff < 0)
				buffer.append(" - " + MathUtil.format(Math.abs(coeff)) + "*" + regressorExpr);
			else
				buffer.append(" + " + MathUtil.format(coeff) + "*" + regressorExpr);
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
	public Object extractResponse(Profile profile) {
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
	 * In the most general case that each index is an mathematical expression, this method is not focused but is useful in some cases.
	 * @param z specified variable Z.
	 * @param inverse if true, there is an inverse transformation.
	 * @return transformed value of Z.
	 */
	protected Object transformResponse(Object z, boolean inverse) {
		// TODO Auto-generated method stub
		return z;
	}

	
	/**
	 * Splitting the specified string into list of indices.
	 * @param cfgIndices specified string.
	 * @return list of indices.
	 */
	public static List<String> splitIndices(String cfgIndices) {
		List<String> txtList = Util.newList();
		if (cfgIndices == null || cfgIndices.isEmpty() || cfgIndices.equals(R_INDICES_FIELD_DEFAULT))
			return txtList;
					
		//The pattern is {1, 2}, {3, 4, 5), {5, 6}, {5, 6, 7, 8}, {9, 10}
		//The pattern can also be 1, 2, 3, 4, 5, 5, 6, 5, 6, 7, 8, 9, 10
		String regex = "\\}(\\s)*,(\\s)*\\{";
		String[] txtArray = cfgIndices.trim().split(regex);
		for (String txt : txtArray) {
			txt = txt.trim().replaceAll("\\}", "").replaceAll("\\{", "");
			if (!txt.isEmpty())
				txtList.add(txt);
		}
		
		return txtList;
	}
	
	
	/**
	 * Parsing indices of variables. In the most general case, each index is an mathematical expression.
	 * @param cfgIndices input configured indices string.
	 * @param maxVariables input maximum variables.
	 * @param xIndicesOutput output regressors indices.
	 * @param zIndicesOutput output response indices.
	 * @return true if parsing is successful.
	 */
	public static boolean parseIndices(String cfgIndices, int maxVariables, List<Object[]> xIndicesOutput, List<Object[]> zIndicesOutput) {
		xIndicesOutput.clear();
		xIndicesOutput.add(new Object[] {new Integer(-1)}); // due to X = (1, x1, x2,..., x(n-1)) and there is no 1 in real data.
		zIndicesOutput.clear();
		zIndicesOutput.add(new Object[] {new Integer(-1)}); // due to Z = (1, z) and there is no 1 in real data.
		
		//Begin extracting indices from configuration.
		//The pattern is {1, 2}, {3, 4, 5), {5, 6}, {5, 6, 7, 8}, {9, 10}
		//The pattern can also be 1, 2, 3, 4, 5, 5, 6, 5, 6, 7, 8, 9, 10
		List<String> txtList = splitIndices(cfgIndices);
		
		List<Object[]> indices = Util.newList();
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
		
		if (indices.size() < 2) { //The case: 1, 2, 3, 4, 5, 5, 6, 5, 6, 7, 8, 9, 10
			for (int j = 0; j < maxVariables - 1; j++)
				xIndicesOutput.add(new Object[] {new Integer(j)});
			zIndicesOutput.add(new Object[] {new Integer(maxVariables - 1)}); //The last index is Z index.
		}
		else {
			for (int j = 0; j < indices.size() - 1; j++)
				xIndicesOutput.add(indices.get(j));
			zIndicesOutput.add(indices.get(indices.size() - 1)); //The last index is Z index
		}
		//End extracting indices from configuration
		
		if (zIndicesOutput.size() < 2 || xIndicesOutput.size() < 2)
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
		List<Object> indices = Util.newList();
		if (txtIndex == null || txtIndex.isEmpty())
			return indices;
		List<String> array = TextParserUtil.split(txtIndex, sep, null);
		
		for (String el : array) {
			if (el.contains(VAR_INDEX_SPECIAL_CHAR)) {
				indices.add(el);
				continue;
			}
			
			int index = -1;
			boolean parseSuccess = true;
			try {
				index = Integer.parseInt(el);
			}
			catch (Throwable e) {
				parseSuccess = false;
			}
			if (parseSuccess && index >= 0)
				indices.add(new Integer(index));
			else
				indices.add(el);
		}
		
		return indices;
	}

	
	/**
	 * Finding object in specified list of indices.
	 * @param indicesList specified list of indices.
	 * @param object specified object.
	 * @return index of specified object in specified list of indices. 
	 */
	public static int findIndex(List<Object[]> indicesList, Object object) {
		for (int i = 0; i < indicesList.size(); i++) {
			Object[] objects = indicesList.get(i);
			for (int j = 0; j < objects.length; j++) {
				if (objects[j].equals(object))
					return i;
			}
		}
		
		return -1;
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
			if (item instanceof Number)
				return profile.getValueAsReal(((Number)item).intValue());
			
			String expr = item.toString().trim();
			int n = profile.getAttCount();
			for (int i = 0; i < n; i++) {
				String attName =  profile.getAtt(i).getName();
				String replacedText = expr.contains(VAR_INDEX_SPECIAL_CHAR) ? VAR_INDEX_SPECIAL_CHAR + attName : attName;   
				if(!expr.contains(replacedText))
					continue;
				
				if(profile.isMissing(i))
					return Constants.UNUSED; //Cannot evaluate
				Double value = profile.getValueAsReal(attName);
				if(!Util.isUsed(value))
					return Constants.UNUSED; //Cannot evaluate
				
				expr = expr.replaceAll(replacedText, value.toString());
			}
			
			Parser parser = new Parser();
			return parser.parse2(expr);
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
		if (item instanceof Number)
			return attList.get(((Number)item).intValue()).getName();
		else {
			String expr = item.toString();
			for (int i = 0; i < attList.size(); i++) {
				String attName =  attList.get(i).getName();
				String replacedText = expr.contains(VAR_INDEX_SPECIAL_CHAR) ? VAR_INDEX_SPECIAL_CHAR + attName : attName;   
				expr = expr.replaceAll(replacedText, attName).trim();
			}
			
			return expr;
		}
	}
	

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
			logger.info("Problem from LU Decomposition: " + e1.getMessage());
			try {
				//Secondly, solve approximate solution by QR Decomposition
				DecompositionSolver solver = new QRDecomposition(M).getSolver(); //It is possible to replace QRDecomposition by LUDecomposition here.
				x = DSUtil.toDoubleList(solver.solve(m).toArray()); //solve Ax = b with approximation
				x = checkSolution(x);
			}
			catch (SingularMatrixException e2) {
				logger.info("Singular matrix problem from QR Decomposition");
				//Finally, solve approximate solution by Moore–Penrose pseudo-inverse matrix
				try {
					DecompositionSolver solver = new SingularValueDecomposition(M).getSolver(); //It is possible to replace QRDecomposition by LUDecomposition here.
					RealMatrix pseudoInverse = solver.getInverse();
					x = DSUtil.toDoubleList(pseudoInverse.operate(m).toArray());
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
	 * Creating default attribute list for sample to learn regression model.
	 * By default, all variables are real numbers.
	 * @param maxVarNumber maximum number of variables.
	 * @return default attribute list for sample to learn regression model.
	 */
	public static AttributeList defaultAttributeList(int maxVarNumber) {
		AttributeList attList = new AttributeList();
		for (int i = 0; i < maxVarNumber; i++) {
			Attribute att = new Attribute("var" + i, Type.real);
			attList.add(att);
		}
		
		return attList;
	}

	
	/**
	 * Extracting real number from specified object.
	 * @param value specified object.
	 * @return real number extracted from specified object.
	 */
	public static double extractNumber(Object value) {
		if (value == null)
			return Constants.UNUSED;
		else if (value instanceof Double)
			return (double)value;
		else if (value instanceof Number)
			return ((Number)value).doubleValue();
		else {
			try {
				Double.parseDouble(value.toString());
			}
			catch (Exception e) {
				e.printStackTrace();
			}
			return Constants.UNUSED;
		}
	}

	
}
