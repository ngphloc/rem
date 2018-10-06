package net.hudup.regression;

import java.awt.Color;
import java.io.BufferedWriter;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;

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

import flanagan.analysis.Regression;
import flanagan.math.Fmath;
import flanagan.plot.PlotGraph;
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
import net.hudup.core.logistic.UriAssoc;
import net.hudup.core.logistic.Vector2;
import net.hudup.core.logistic.xURI;
import net.hudup.core.parser.TextParserUtil;
import net.hudup.regression.em.ui.graph.Graph;
import net.hudup.regression.em.ui.graph.PlotGraphExt;

/**
 * This is the most abstract class for regression model. It implements partially the interface {@link RM}.
 * 
 * @author Loc Nguyen
 * @version 1.0*
 */
public abstract class AbstractRM extends AbstractTestingAlg implements RM2 {

	
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
	public AbstractRM() {
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
		if (!AbstractRM.parseIndices(cfgIndices, profile0.getAttCount(), this.xIndices, this.zIndices))
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
			
			double lastValue = (double)extractResponseValue(profile);
			if (Util.isUsed(lastValue))
				zExists = zExists || true; 
			
			for (int j = 1; j < xIndices.size(); j++) {
				double value = extractRegressorValue(profile, j);
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
		if (this.coeffs == null || input == null)
			return null;
		
		Profile profile = null;
		if (input instanceof Profile)
			profile = (Profile)input;
		else
			profile = createProfile(this.attList, input);
		if (profile == null)
			return null;
		
		double sum = this.coeffs.get(0);
		for (int j = 1; j < this.coeffs.size(); j++) {  //due to x = (1, x1, x2,..., xn) and xIndices.get(0) = -1
			double value = extractRegressorValue(profile, j);
			sum += this.coeffs.get(j) * (double)transformRegressor(value, false); 
		}
		
		return transformResponse(sum, true);
	}

	
	/**
	 * Executing this algorithm by arbitrary input parameter.
	 * @param input arbitrary input parameter.
	 * @return result of execution. Return null if execution is failed.
	 */
	public Object executeIntel(Object...input) {
		return execute(input);
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
		config.put(R_INDICES_FIELD, R_INDICES_DEFAULT);
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
		buffer.append(transformResponse(extractResponse().toString(), false) + " = " + MathUtil.format(coeffs.get(0)));
		for (int j = 0; j < this.coeffs.size() - 1; j++) {
			double coeff = this.coeffs.get(j + 1);
			String regressorExpr = "(" + transformRegressor(extractRegressor(j + 1).toString(), false).toString() + ")";
			if (coeff < 0)
				buffer.append(" - " + MathUtil.format(Math.abs(coeff)) + "*" + regressorExpr);
			else
				buffer.append(" + " + MathUtil.format(coeff) + "*" + regressorExpr);
		}
		
		return buffer.toString();
	}
	
	
	@Override
	public VarWrapper extractRegressor(int index) {
		// TODO Auto-generated method stub
		return extractVariable(attList, xIndices, index);
	}

	
	@Override
	public List<VarWrapper> extractRegressors() {
		// TODO Auto-generated method stub
		return extractVariables(attList, xIndices);
	}


	@Override
	public List<VarWrapper> extractSingleRegressors() {
		// TODO Auto-generated method stub
		return extractSingleVariables(attList, xIndices);
	}


	@Override
	public double extractRegressorValue(Object input, int index) {
		// TODO Auto-generated method stub
		if (input == null)
			return Constants.UNUSED;
		else if (input instanceof Profile)
			return extractVariableValue(input, null, xIndices, index);
		else
			return extractVariableValue(input, attList, xIndices, index);
	}


	@Override
	public VarWrapper extractResponse() {
		// TODO Auto-generated method stub
		return extractVariable(attList, zIndices, 1);
	}


	@Override
	public synchronized Object extractResponseValue(Object input) {
		// TODO Auto-generated method stub
		if (input == null)
			return Constants.UNUSED;
		else if (input instanceof Profile)
			return extractVariableValue(input, null, zIndices, 1);
		else
			return extractVariableValue(input, attList, zIndices, 1);
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


	@Override
	public Object transformResponse(Object z, boolean inverse) {
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
		if (cfgIndices == null || cfgIndices.isEmpty() || cfgIndices.equals(R_INDICES_DEFAULT))
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
			if (parseSuccess)
				indices.add(new Integer(index - 1)); //Index begins 1. Please pay attention to this line.
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
	 * Creating profile from specified attribute list and object.
	 * @param attList specified attribute list
	 * @param object specified object.
	 * @return profile from specified attribute list and object.
	 */
	public static Profile createProfile(AttributeList attList, Object object) {
		if (attList == null || attList.size() == 0)
			return null;
		
		List<Double> values = null;
		Map<String, Object> mapValues = null;
		if (object == null)
			values = Util.newList();
		else {
			if (object instanceof Map<?, ?>) {
				Map<?, ?> map = (Map<?, ?>)object;
				Set<?> keys = map.keySet();
				mapValues = Util.newMap();
				for (Object key : keys) {
					mapValues.put(key.toString(), map.get(key));
				}
			}
			else
				values = DSUtil.toDoubleList(object, false);
		}
		
		Profile profile = new Profile(attList);
		if (values != null) {
			int n = Math.min(values.size(), attList.size());
			for (int j = 0; j < n; j++) {
				int start = values.size() > attList.size() ? 1 : 0;
				profile.setValue(j, values.get(j + start));
			}
		}
		else if (mapValues != null) {
			Set<String> keys = mapValues.keySet();
			for (String key : keys) {
				profile.setValue(key, mapValues.get(key));
			}
		}
		
		return profile;
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

	
	/**
	 * Testing whether the deviation between estimated value and current value is not satisfied a threshold.
	 * @param estimatedValue estimated value.
	 * @param currentValue current value.
	 * @param threshold specified threshold.
	 * @return true if the deviation between estimated value and current value is not satisfied a threshold.
	 */
	public static boolean notSatisfy(double estimatedValue, double currentValue, double threshold) {
		return Math.abs(estimatedValue - currentValue) > threshold * Math.abs(currentValue);
	}
	
	
	/**
	 * Extracting variable.
	 * @param attList specified attribute list.
	 * @param indices specified list of indices.
	 * @param index specified index. Index 0 is not included in the profile because this specified index is in the parameter <code>indices</code>.
	 * So index 0 always indicate to value &apos;#noname&apos;. 
	 * @return variable.
	 */
	public static VarWrapper extractVariable(AttributeList attList, List<Object[]> indices, int index) {
		// TODO Auto-generated method stub
		if (index == 0 || attList == null) return null;

		VarWrapper var = null;
		Object item = indices.get(index)[0];
		if (item instanceof Number) {
			int attIndex = ((Number)item).intValue();
			var = VarWrapper.createByName(index, attList.get(attIndex).getName());
			var.setAttribute(attList.get(attIndex));
		}
		else {
			String expr = item.toString();
			for (int j = 0; j < attList.size(); j++) {
				String attName =  attList.get(j).getName();
				String replacedText = expr.contains(VAR_INDEX_SPECIAL_CHAR) ? VAR_INDEX_SPECIAL_CHAR + attName : attName;   
				expr = expr.replaceAll(replacedText, attName).trim();
			}
			
			var = VarWrapper.createByExpr(index, expr);
		}
		
		return var;
	}

	
	/**
     * Getting list of variables from specified indices and attribute list.
     * @param attList specified attribute list.
     * @param indices specified indices.
     * @return list of variables.
     */
    public static List<VarWrapper> extractVariables(AttributeList attList, List<Object[]> indices) {
    	List<VarWrapper> vars = Util.newList();
    	if (indices == null || indices.size() <= 1)
    		return vars;
    	
    	for (int i = 1; i < indices.size(); i++) {
			Object item = indices.get(i)[0];
			VarWrapper var = null;
			if (item instanceof Number) {
				int attIndex = ((Number)item).intValue();
				var = VarWrapper.createByName(i, attList.get(attIndex).getName());
				var.setAttribute(attList.get(attIndex));
			}
			else {
				String expr = item.toString().trim();
				for (int j = 0; j < attList.size(); j++) {
					String attName =  attList.get(j).getName();
					String replacedText = expr.contains(VAR_INDEX_SPECIAL_CHAR) ? VAR_INDEX_SPECIAL_CHAR + attName : attName;   
					expr = expr.replaceAll(replacedText, attName).trim();
				}
				var = VarWrapper.createByExpr(i, expr);
			}
			
			vars.add(var);
    	}
    	
    	return vars;
    }


	/**
     * Getting list of actual variables from specified indices and attribute list.
     * @param attList specified attribute list.
     * @param indices specified indices.
     * @return list of variables.
     */
    public static List<VarWrapper> extractSingleVariables(AttributeList attList, List<Object[]> indices) {
    	List<VarWrapper> vars = Util.newList();
    	if (indices == null || attList == null) return vars;
    	
    	for (int j = 0; j < attList.size(); j++) {
    		Attribute att = attList.get(j);
    		
    		boolean found = false;
    		int foundIndex = -1;
        	for (int i = 1; i < indices.size(); i++) {
    			Object item = indices.get(i)[0];
    			if (item instanceof Number) {
    				if (((Number)item).intValue() == j) {
    					found = true;
    					foundIndex = i;
    					break;
    				}
    			}
    			else {
    				String expr = item.toString().trim();
    				String replacedText = expr.contains(VAR_INDEX_SPECIAL_CHAR) ? VAR_INDEX_SPECIAL_CHAR + att.getName() : att.getName();   
    				if(expr.contains(replacedText)) {
    					found = true;
    					break;
    				}
    			}
        	}
        	
        	if (found) {
        		VarWrapper var = VarWrapper.createByName(foundIndex, att.getName());
        		var.setAttribute(att);
        		vars.add(var);
        	}
    	}
    	
    	return vars;
    }
    
    
	/**
	 * Extracting value of variable (X) from specified profile.
	 * @param input specified input. It is often a profile.
	 * @param attList specified attribute list.
	 * @param indices specified list of indices.
	 * @param index specified index. Index 0 is not included in the profile because this specified index is in the parameter <code>indices</code>.
	 * So index 0 always indicate to value 1. 
	 * @return value of variable (X) extracted from specified profile.
	 */
	public static double extractVariableValue(Object input, AttributeList attList, List<Object[]> indices, int index) {
		if (index == 0)
			return 1.0;
		if (input == null)
			return Constants.UNUSED;
		
		if (!(input instanceof Profile)) {
			List<Double> values = DSUtil.toDoubleList(input, false);
			if (attList == null)
				attList = defaultAttributeList(values.size());
			Profile profile = createProfile(attList, values);
			return extractVariableValue(profile, null, indices, index);
		}
		
		Profile profile = (Profile)input;
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
     * Creating graph for response variable given large statistic and regression model.
     * @param rm given regression model.
     * @param stats given large statistic.
     * @return graph for response variable.
     */
    public static Graph createResponseGraph(RM2 rm, LargeStatistics stats) {
		if (rm == null || stats == null)
			return null;
		
    	int ncurves = 2;
    	int npoints = stats.size();
    	double[][] data = PlotGraph.data(ncurves, npoints);

    	for(int i = 0; i < npoints; i++) {
            data[0][i] = (double)rm.transformResponse(stats.getZData().get(i)[1], true);
            data[1][i] = rm.executeByXStatistic(stats.getXData().get(i));
        }

    	Regression regression = new Regression(data[0], data[1]);
    	regression.linear();
    	double[] coef = regression.getCoeff();
    	data[2][0] = Fmath.minimum(data[0]);
    	data[3][0] = coef[0] + coef[1] * data[2][0];
    	data[2][1] = Fmath.maximum(data[0]);
    	data[3][1] = coef[0] + coef[1] * data[2][1];

    	PlotGraphExt pg = new PlotGraphExt(data) {

			/**
			 * Serial version UID for serializable class.
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public String getGraphFeature() {
				// TODO Auto-generated method stub
				return "R=" + MathUtil.format(rm.calcR(), 2);
			}
    		
    	};

    	pg.setGraphTitle("Correlation plot: " + pg.getGraphFeature());
    	pg.setXaxisLegend("Real " + rm.transformResponse(rm.extractResponse().toString(), true));
    	pg.setYaxisLegend("Estimated " + rm.transformResponse(rm.extractResponse().toString(), true));
    	int[] popt = {1, 0};
    	pg.setPoint(popt);
    	int[] lopt = {0, 3};
    	pg.setLine(lopt);

    	pg.setBackground(Color.WHITE);
        return pg;
    }


    /**
     * Creating error graph for response variable given regression model and large statistics.
     * @param rm given regression model.
     * @param stats given large statistics.
     * @return error graph for response variable.
     */
    public static Graph createErrorGraph(RM2 rm, LargeStatistics stats) {
		if (rm == null || stats == null)
			return null;
    	
    	int ncurves = 4;
    	int npoints = stats.size();
    	double[][] data = PlotGraph.data(ncurves, npoints);

		double errorMean = 0;
    	for(int i = 0; i < npoints; i++) {
            double z = (double)rm.transformResponse(stats.getZData().get(i)[1], true);
            double zEstimated = rm.executeByXStatistic(stats.getXData().get(i));
            data[0][i] = ( z + zEstimated ) / 2.0;
            data[1][i] = zEstimated - z;
            
            errorMean += data[1][i];
        }
    	errorMean = errorMean / npoints;
    	double errorSd = 0;
    	for(int i = 0; i < npoints; i++) {
    		double d = data[1][i] - errorMean;
    		errorSd += d*d;
    	}
   		errorSd = Math.sqrt(errorSd / npoints); //MLE estimation
    		
    	// Mean - 1.96sd
    	data[2][0] = 0;
    	data[3][0] = errorMean - 1.96 * errorSd;
    	data[2][1] = Fmath.maximum(data[0]);
    	data[3][1] = errorMean - 1.96 * errorSd;

    	// Mean
    	data[4][0] = 0;
    	data[5][0] = errorMean;
    	data[4][1] = Fmath.maximum(data[0]);
    	data[5][1] = errorMean;

    	// Mean + 1.96sd
    	data[6][0] = 0;
    	data[7][0] = errorMean + 1.96 * errorSd;
    	data[6][1] = Fmath.maximum(data[0]);
    	data[7][1] = errorMean + 1.96 * errorSd;

    	final double mean = errorMean, sd = errorSd;
    	PlotGraphExt pg = new PlotGraphExt(data) {

			/**
			 * Serial version UID for serializable class.
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public String getGraphFeature() {
				// TODO Auto-generated method stub
				return MathUtil.format(mean, 2) + " +/- 1.96*" + 
    				MathUtil.format(sd, 2);
			}
    		
    	};

    	pg.setGraphTitle("Error plot: " + pg.getGraphFeature());
    	pg.setXaxisLegend("Mean " + rm.transformResponse(rm.extractResponse().toString(), true));
    	pg.setYaxisLegend("Estimated error");
    	int[] popt = {1, 0, 0, 0};
    	pg.setPoint(popt);
    	int[] lopt = {0, 3, 3, 3};
    	pg.setLine(lopt);

    	pg.setBackground(Color.WHITE);
    	
        return pg;
    }


    /**
     * Creating graph related to response variable given regression model.
     * @param rm given regression model.
     * @return graphs related to response variable given regression model.
     */
    public static List<Graph> createResponseRalatedGraphs(RM2 rm) {
    	List<Graph> relatedGraphs = Util.newList();
    	
    	Graph responseGraph = rm.createResponseGraph();
    	if (responseGraph != null) relatedGraphs.add(responseGraph);
    	
    	Graph errorGraph = rm.createErrorGraph();
    	if (errorGraph != null) relatedGraphs.add(errorGraph);

    	return relatedGraphs;
    }


    /**
     * Calculating variance with specified regression model and large statistics.
     * @param rm specified regression model.
     * @param stats specified large statistics.
     * @return variance with specified regression model and large statistics.
     */
	public static double calcVariance(RM2 rm, LargeStatistics stats) {
		// TODO Auto-generated method stub
		if (rm == null || stats == null)
			return Constants.UNUSED;
		
		List<double[]> xData = stats.getXData();
		List<double[]> zData = stats.getZData();
		
		double ss = 0;
		int N = 0;
		for (int i = 0; i < xData.size(); i++) {
			double[] xVector = xData.get(i);
			double z = (double)rm.transformResponse(zData.get(i)[1], true);
			double zEstimated = rm.executeByXStatistic(xVector);
			
			if (Util.isUsed(z) && Util.isUsed(zEstimated)) {
				ss += (zEstimated - z) * (zEstimated - z);
				N++;
			}
		}
		return ss / N;
	}


    /**
     * Calculating correlation with specified regression model and large statistics.
     * @param rm specified regression model.
     * @param stats specified large statistics.
     * @return correlation with specified regression model and large statistics.
     */
	public static double calcR(RM2 rm, LargeStatistics stats) {
		// TODO Auto-generated method stub
		if (rm == null || stats == null)
			return Constants.UNUSED;
		
		Vector2 zVector = new Vector2(stats.size(), 0);
		Vector2 zEstimatedVector = new Vector2(stats.size(), 0);
		for (int i = 0; i < stats.size(); i++) {
            double z = (double)rm.transformResponse(stats.getZData().get(i)[1], true);
            zVector.set(i, z);
            
            double zEstimated = rm.executeByXStatistic(stats.getXData().get(i));
            zEstimatedVector.set(i, zEstimated);
		}
		
		return zEstimatedVector.corr(zVector);
	}


    /**
     * Calculating error with specified regression model and large statistics.
     * @param rm specified regression model.
     * @param stats specified large statistics.
     * @return error with specified regression model and large statistics.
     */
	public static double[] calcError(RM2 rm, LargeStatistics stats) {
		// TODO Auto-generated method stub
		if (rm == null || stats == null)
			return null;
		
		Vector2 error = new Vector2(stats.size(), 0);
		for (int i = 0; i < stats.size(); i++) {
            double z = (double)rm.transformResponse(stats.getZData().get(i)[1], true);
            double zEstimated = rm.executeByXStatistic(stats.getXData().get(i));
            error.set(i, zEstimated - z);
		}
		
    	return new double[] {error.mean(), error.mleVar()};
	}


	/**
	 * Saving large statistics at specified URI.
	 * @param rm specified regression model.
	 * @param stats specified large statistics.
	 * @param uri specified URI.
	 * @param decimal specified decimal.
	 * @return true if saving is successful.
	 */
	public static boolean saveLargeStatistics(RM2 rm, LargeStatistics stats, xURI uri, int decimal) {
		// TODO Auto-generated method stub
		if (rm == null || stats == null || stats.size() == 0 || uri == null)
			return false;
		
		UriAssoc uriAssoc = Util.getFactory().createUriAssoc(uri);
		if (uriAssoc == null) return false;
		
		try {
			BufferedWriter writer = new BufferedWriter(uriAssoc.getWriter(uri, false));
			
			StringBuffer columns = new StringBuffer();
			List<VarWrapper> regressors = rm.extractRegressors();
			for (int i = 0; i < regressors.size(); i++) {
				VarWrapper regressor = regressors.get(i);
				if (i > 0)
					columns.append(", ");
				columns.append(regressor.toString());
			}
			VarWrapper response = rm.extractResponse();
			columns.append(", " + response.toString());
			writer.write(columns.toString());
			
			for (int i = 0; i < stats.size(); i++) {
				double[] xVector = stats.getXData().get(i);
				double[] zVector = stats.getZData().get(i);
				StringBuffer row = new StringBuffer(xVector.length + 1);
				
				row.append("\n");
				for (int j = 1; j < xVector.length; j++) {
					if (decimal > 0)
						row.append(MathUtil.format(xVector[j], decimal));
					else
						row.append(MathUtil.format(xVector[j]));
					row.append(", ");
				}
				if (decimal > 0)
					row.append(MathUtil.format(zVector[1], decimal));
				else
					row.append(MathUtil.format(zVector[1]));
				
				writer.write(row.toString());
			}

			writer.close();
			return true;
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		
		return false;
	}


}
