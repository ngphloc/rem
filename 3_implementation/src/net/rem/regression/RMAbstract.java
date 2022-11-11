/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.regression;

import java.awt.Color;
import java.io.Writer;
import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import flanagan.analysis.Regression;
import flanagan.math.Fmath;
import flanagan.plot.PlotGraph;
import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.ExecutableAlgAbstract;
import net.hudup.core.alg.MemoryBasedAlg;
import net.hudup.core.alg.MemoryBasedAlgRemote;
import net.hudup.core.data.Attribute;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Dataset;
import net.hudup.core.data.Fetcher;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.DSUtil;
import net.hudup.core.logistic.Inspector;
import net.hudup.core.logistic.LogUtil;
import net.hudup.core.logistic.MathUtil;
import net.hudup.core.logistic.UriAssoc;
import net.hudup.core.logistic.Vector2;
import net.hudup.core.logistic.xURI;
import net.hudup.core.logistic.ui.UIUtil;
import net.hudup.core.parser.TextParserUtil;
import net.rem.regression.em.ui.REMInspector;
import net.rem.regression.em.ui.graph.Graph;
import net.rem.regression.em.ui.graph.PlotGraphExt;
import net.rem.regression.logistic.speqmath.Parser;

/**
 * This is the most abstract class for regression model. It implements partially the interface {@link RM}.
 * 
 * @author Loc Nguyen
 * @version 1.0*
 */
public abstract class RMAbstract extends ExecutableAlgAbstract implements RM, RMRemote, MemoryBasedAlg, MemoryBasedAlgRemote {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Regression coefficient.
	 */
	protected List<Double> coeffs = null;
	
	
	/**
	 * Indices for X data, including -1 such as (-1, 0 or #x1, 1 or log(#x1), ..., n-1 or xn). So x1 begins with index 1.
	 * Note, every element of xIndices is an array of objects. In current implementation, only the first object in such array is used, which can be index (the index -1 points to 1 value, not point to profile) or expression.
	 */
	protected List<Object[]> xIndices = Util.newList();

	
	/**
	 * Indices for Z data, including -1 such as (-1, z). So z has index 1.
	 * Note, every element of zIndices is an array of objects. In current implementation, only the first object in such array is used, which can be index (the index -1 points to 1 value, not point to profile) or expression.
	 * However, zIndices has 2 elements.
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
	public RMAbstract() {
		super();
	}
	
	
	@Override
	protected Object fetchSample(Dataset dataset) {
		return dataset != null ? dataset.fetchSample() : null;
	}

	
	@Override
	public synchronized Object learnStart(Object...info) throws RemoteException {
		UsedIndices usedIndices = UsedIndices.extract(info);
		boolean prepared = usedIndices != null ? prepareInternalData(usedIndices.xIndicesUsed, usedIndices.zIndicesUsed) : prepareInternalData();
			
		Object resulted = null;
		if (prepared)
			resulted = learn0();
		if (resulted == null)
			clearInternalData();
		
		return resulted;
	}

	
	/**
	 * Internal learning parameters. Derived class needs to implement this method.
	 * @return the parameter to be learned.
	 * @throws RemoteException if any error occurs.
	 */
	protected abstract Object learn0() throws RemoteException;
	
	
	/**
	 * Preparing data.
	 * @param xIndicesUsed indicator of used X indices (xIndices).
	 * For xIndices, regressors begin from 1 due to X = (1, x1, x2,..., x(n-1)) and so, the first element (0) of this indices array is -1 pointing to 1 value.
	 * Therefore, xIndicesUsed[0] is always 0.
	 * @param zIndicesUsed indicator of used Z indices.
	 * For zIndices, due to Z = (1, z), the first element (0) of this indices array is -1 pointing to 1 value.
	 * Therefore, zIndicesUsed[0] is always 0.
	 * @return true if data preparation is successful.
	 * @throws RemoteException if any error raises.
	 */
	@SuppressWarnings("unchecked")
	protected boolean prepareInternalData(int[] xIndicesUsed, int[] zIndicesUsed) throws RemoteException {
		clearInternalData();
		
		Profile profile0 = null;
		if (((Fetcher<Profile>)sample).next()) {
			profile0 = ((Fetcher<Profile>)sample).pick();
		}
		((Fetcher<Profile>)sample).reset();
		if (profile0 == null)
			return false;
		if (profile0.getAttCount() < 2) //x1, x2,..., x(n-1), z
			return false;
		this.attList = profile0.getAttRef();
		
		//Begin parsing indices
		String cfgIndices = this.getConfig().getAsString(R_INDICES_FIELD);
		if (!RMAbstract.parseIndices(cfgIndices, profile0.getAttCount(), this.xIndices, this.zIndices))
			return false;
		//End parsing indices

		//Begin adjusting indices
		this.xIndices = RMAbstract.extractIndices(this.xIndices, xIndicesUsed);
		this.zIndices = RMAbstract.extractIndices(this.zIndices, zIndicesUsed);
		//End adjusting indices
		
		//Begin checking existence of values.
		boolean zExists = false;
		boolean[] xExists = new boolean[xIndices.size() - 1]; //profile = (x1, x2,..., x(n-1), z)
		Arrays.fill(xExists, false);
		while (((Fetcher<Profile>)sample).next()) {
			Profile profile = ((Fetcher<Profile>)sample).pick(); //profile = (x1, x2,..., x(n-1), z)
			if (profile == null)
				continue;
			
			double lastValue = (double)extractResponseValue0(profile);
			if (Util.isUsed(lastValue))
				zExists = zExists || true; 
			
			for (int j = 1; j < xIndices.size(); j++) {
				double value = extractRegressorValue0(profile, j);
				if (Util.isUsed(value))
					xExists[j - 1] = xExists[j - 1] || true;
			}
		}
		((Fetcher<Profile>)sample).reset();

		List<Object[]> xIndicesTemp = Util.newList();
		xIndicesTemp.add(xIndices.get(0)); //adding -1
		for (int j = 1; j < xIndices.size(); j++) {
			if (xExists[j - 1])
				xIndicesTemp.add(xIndices.get(j)); //only use variables having at least one value.
		}
		if (!zExists || xIndicesTemp.size() < 2) //Please pay attention here.
			return false;
		xIndices = xIndicesTemp;
		//End checking existence of values.

		return true;
	}

	
	/**
	 * Preparing data.
	 * @return true if data preparation is successful.
	 * @throws RemoteException if any error raises.
	 */
	protected boolean prepareInternalData() {
		try {
			return prepareInternalData(null, null);
		} catch (Throwable e) {LogUtil.trace(e);}
		
		return false;
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
	public synchronized Object execute(Object input) throws RemoteException {
		if (this.coeffs == null || input == null)
			return null;
		
		Profile profile = null;
		if (input instanceof Profile)
			profile = (Profile)input;
		else
			profile = Profile.createProfile(this.attList, input);
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
	 * @throws RemoteException if any error raises.
	 */
	public Object executeIntel(Object...input) throws RemoteException {
		return execute(input);
	}

	
	@Override
	public synchronized Object getParameter() throws RemoteException {
		return coeffs;
	}

	
	@Override
	public String parameterToShownText(Object parameter, Object... info) throws RemoteException {
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
	public synchronized String getDescription() throws RemoteException {
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
	public synchronized Inspector getInspector() {
		return getInspector(this);
	}


	/**
	 * Getting inspector of regression model.
	 * @param rm specified regression model.
	 * @return inspector of regression model.
	 */
	public static Inspector getInspector(RM rm) {
		Object parameter = null;
		try {
			parameter = rm.getParameter();
		} catch (Exception e) {
			LogUtil.trace(e);
			parameter = null;
		}
		
		if (parameter == null) {
			LogUtil.error("Invalid regression model");
			return null;
		}
		else {
			try {
				return new REMInspector(UIUtil.getDialogForComponent(null), rm);
			} 
			catch (Exception e) {
				LogUtil.trace(e);
				LogUtil.error("Cannot retrieve inspector");
			}
			
			return null;
		}
	}

	
	@Override
	public String[] getBaseRemoteInterfaceNames() throws RemoteException {
		return new String[] {RMRemote.class.getName(), MemoryBasedAlgRemote.class.getName()};
	}

	
	@Override
	public VarWrapper extractRegressor(int index) throws RemoteException {
		return extractVariable(attList, xIndices, index);
	}

	
	@Override
	public List<VarWrapper> extractRegressors() throws RemoteException {
		return extractVariables(attList, xIndices);
	}


	@Override
	public List<VarWrapper> extractSingleRegressors() throws RemoteException {
		return extractSingleVariables(attList, xIndices);
	}


	@Override
	public double extractRegressorValue(Object input, int index) throws RemoteException {
		return extractRegressorValue0(input, index);
	}


	/**
	 * Extracting value of regressor (X) from specified profile.
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @param input specified input. It is often profile. It can be an array of real values.
	 * @param index specified index. Index 0 is not included in the profile because this specified index is in the parameter r_indices.
	 * So the index here is the second index, and of course it is number.
	 * Index starts from 1. So index 0 always indicates to value 1. 
	 * @return value of regressor (X) extracted from specified profile. Note, the returned value is not transformed.
	 */
	private double extractRegressorValue0(Object input, int index) {
		if (input == null)
			return Constants.UNUSED;
		else if (input instanceof Profile)
			return extractVariableValue(input, null, xIndices, index);
		else
			return extractVariableValue(input, attList, xIndices, index);
	}

	
	@Override
	public double[] extractRegressorValues(Object input) throws RemoteException {
		return extractVariableValues(input, attList, xIndices);
	}


	@Override
	public VarWrapper extractResponse() throws RemoteException {
		return extractVariable(attList, zIndices, 1);
	}


	@Override
	public Object extractResponseValue(Object input) throws RemoteException {
		return extractResponseValue0(input);
	}


	/**
	 * Extracting value of response variable (Z) from specified profile.
	 * In the most general case that each index is an mathematical expression, this method is focused.
	 * @param input specified input. It is often profile but it can be an array of real values.
	 * @return value of response variable (Z) extracted from specified profile.
	 */
	private Object extractResponseValue0(Object input) {
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
		return x;
	}


	@Override
	public Object transformResponse(Object z, boolean inverse) throws RemoteException {
		return z;
	}

	
	@Override
	public DataConfig createDefaultConfig() {
		DataConfig config = super.createDefaultConfig();
		config.put(R_INDICES_FIELD, R_INDICES_DEFAULT);
		return config;
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
	 * @param xIndicesOutput output regressors indices. Regressors begin from 1 due to X = (1, x1, x2,..., x(n-1)) and so,
	 * the first element (0) of this indices array is -1 pointing to 1 value. 
	 * @param zIndicesOutput output response indices. Due to Z = (1, z), the first element (0) of this indices array is -1 pointing to 1 value.
	 * @return true if parsing is successful.
	 */
	public static boolean parseIndices(String cfgIndices, int maxVariables, List<Object[]> xIndicesOutput, List<Object[]> zIndicesOutput) {
		xIndicesOutput.clear();
		xIndicesOutput.add(new Object[] {Integer.valueOf(-1)}); // due to X = (1, x1, x2,..., x(n-1)) and there is no 1 in real data.
		zIndicesOutput.clear();
		zIndicesOutput.add(new Object[] {Integer.valueOf(-1)}); // due to Z = (1, z) and there is no 1 in real data.
		
		//Begin extracting indices from configuration.
		//The pattern is {#x1, 2}, {3, 4, 5), {log(#x5), 6}, {5, 6, 7, 8}, {9, 10}
		//The pattern can also be 1, 2, 3, 4, 5, 5, 6, 5, 6, 7, 8, 9, 10
		List<String> txtList = splitIndices(cfgIndices);
		
		List<Object[]> indices = Util.newList();
		if (txtList.size() == 1) { //The case: 1, 2, 3, #x4, 5, 5, 6, 5, 6, 7, 8, 9, 10
			List<Object> oneIndices = parseIndex(txtList.get(0), ",");
			for (Object index : oneIndices)
				indices.add(new Object[] {index});
		}
		else if (txtList.size() > 1) { //The case: {#x1, 2}, {3, 4, 5), {log(#x5), 6}, {5, 6, 7, 8}, {9, 10}
			for (String txt : txtList) {
				List<Object> oneIndices = parseIndex(txt, ",");
				if (oneIndices.size() == 0)
					continue;
				indices.add(oneIndices.toArray());
			}
		}
		
		if (indices.size() < 2) { //The case: 1, 2, 3, #x4, 5, 5, 6, 5, 6, 7, 8, 9, 10
			for (int j = 0; j < maxVariables - 1; j++)
				xIndicesOutput.add(new Object[] {Integer.valueOf(j)});
			zIndicesOutput.add(new Object[] {Integer.valueOf(maxVariables - 1)}); //The last index is Z index.
		}
		else { //The case: {#x1, 2}, {3, 4, 5), {log(#x5), 6}, {5, 6, 7, 8}, {9, 10}
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
				indices.add(Integer.valueOf(index - 1)); //Index begins 1. Please pay attention to this line.
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
	 * Creating used sequence of indices.
	 * @param maxIndices the number of elements in X indices (xIndices) or Z indices (zIndices).
	 * For xIndices, regressors begin from 1 due to X = (1, x1, x2,..., x(n-1)) and so, the first element (0) of this indices array is -1 pointing to 1 value. 
	 * For zIndices, due to Z = (1, z), the first element (0) of this indices array is -1 pointing to 1 value.
	 * Therefore, this parameter maxIndices is the length of xIndices or zIndices.
	 * @param used indicating existent indices. Therefore, used[0] is always 0.
	 * @return bit set as used sequence.
	 */
	public static BitSet usedIndicesToBitset(int maxIndices, int[] used) {
		if (maxIndices <= 0) return null;
		BitSet bsFull = new BitSet(maxIndices); bsFull.set(0, maxIndices);
		if (used == null || used.length < 2 || used[0] != 0) return bsFull;
		
		BitSet bs = new BitSet(maxIndices);
		for (int pos : used) {
			if (pos >= 0 && pos < maxIndices) bs.set(pos);
		}
		
		return bs.cardinality() >= 2 ? bs : bsFull;
	}
	
	
	/**
	 * Converting bit set to used indices.
	 * @param orginalIndices original indices.
	 * For X indices (xIndices), regressors begin from 1 due to X = (1, x1, x2,..., x(n-1)) and so, the first element (0) of this indices array is -1 pointing to 1 value. 
	 * For Z indices (zIndices), due to Z = (1, z), the first element (0) of this indices array is -1 pointing to 1 value.
	 * @param bs bit set pointing to used indices. Therefore, bs[0] is always 1.
	 * @return indices extracted from original indices and used bit set.
	 */
	public static int[] bitsetToUsedIndices(List<Object[]> originalIndices, BitSet bs) {
		if (originalIndices == null || originalIndices.size() < 2) return null;
		
		int[] full = new int[originalIndices.size()];
		for (int i = 0; i < originalIndices.size(); i++) full[i] = i;
		if (bs == null || bs.cardinality() < 2) return full;
		
		List<Integer> list = Util.newList();
		for (int i = bs.nextSetBit(0); i >= 0; i = bs.nextSetBit(i + 1)) list.add(i);
		
		int[] used = DSUtil.toIntArray(list);
		return used[0] == 0 ? used : full;
	}
	
	
	
	/**
	 * Extracting used indices.
	 * @param indices original indices.
	 * For xIndices, regressors begin from 1 due to X = (1, x1, x2,..., x(n-1)) and so, the first element (0) of this indices array is -1 pointing to 1 value. 
	 * For zIndices, due to Z = (1, z), the first element (0) of this indices array is -1 pointing to 1 value.
	 * @param used indicating existent indices. Therefore, used[0] is always 0.
	 * @return used indices.
	 */
	public static List<Object[]> extractIndices(List<Object[]> indices, int[] used) {
		if (indices == null || indices.size() < 2 || used == null || used.length < 2 || used[0] != 0)
			return indices;
		
		List<Object[]> newIndices = Util.newList();
		for (int idx : used) {
			if (idx >= 0 && idx < indices.size()) newIndices.add(indices.get(idx));
		}

		return newIndices.size() >= 2 ? newIndices : indices;
	}
	
	
	/**
	 * This class represents used indices.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	public static class UsedIndices {
		
		/**
		 * Indicator of used X indices (xIndices).
		 * For xIndices, regressors begin from 1 due to X = (1, x1, x2,..., x(n-1)) and so, the first element (0) of this indices array is -1 pointing to 1 value.
		 * Therefore, xIndicesUsed[0] is always 0.
		 */
		public int[] xIndicesUsed = null;
		
		/**
		 * Indicator of used Z indices.
		 * For zIndices, due to Z = (1, z), the first element (0) of this indices array is -1 pointing to 1 value.
		 * Therefore, zIndicesUsed[0] is always 0.
		 */
		public int[] zIndicesUsed = null;
		
		/**
		 * Constructor with Indicator of used X indices (xIndices) and indicator of used Z indices.
		 * @param xIndicesUsed Indicator of used X indices (xIndices).
		 * @param zIndicesUsed Indicator of used Z indices.
		 */
		public UsedIndices(int[] xIndicesUsed, int[] zIndicesUsed) {
			this.xIndicesUsed = xIndicesUsed;
			this.zIndicesUsed = zIndicesUsed;
		}
		
		/**
		 * Extracting used indices from specified information.
		 * @param info specified information.
		 * @return used indices extracted from specified information.
		 */
		public static UsedIndices extract(Object...info) {
			if (info == null || info.length == 0) return null;
			for (Object object : info) {
				if (object instanceof UsedIndices) return (UsedIndices)object;
			}
			
			return null;
		}
		
	}
	
	
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
				LogUtil.trace(e);
			}
			return Constants.UNUSED;
		}
	}

	
	/**
	 * Testing whether the deviation between estimated value and current value is not satisfied a threshold.
	 * @param estimatedValue estimated value.
	 * @param currentValue current value.
	 * @param threshold specified threshold.
	 * @param ratioMode flag to indicate whether the threshold is for ratio.
	 * @return true if the deviation between estimated value and current value is not satisfied a threshold.
	 */
	public static boolean notSatisfy(double estimatedValue, double currentValue, double threshold, boolean ratioMode) {
		if (ratioMode)
			return Math.abs(estimatedValue - currentValue) > threshold * Math.abs(currentValue);
		else
			return Math.abs(estimatedValue - currentValue) > threshold;
	}
	
	
	/**
	 * Extracting variable.
	 * @param attList specified attribute list.
	 * @param indices specified list of indices.
	 * @param index specified index. Index 0 is not included in the profile because this specified index is in the parameter <code>indices</code>.
	 * So index 0 always indicate to value #noname.
	 * @return variable.
	 */
	public static VarWrapper extractVariable(AttributeList attList, List<Object[]> indices, int index) {
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
	 * Extracting value of variable (X) from specified input which is often a profile.
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
				attList = AttributeList.defaultRealVarAttributeList(values.size());
			Profile profile = Profile.createProfile(attList, values);
			return extractVariableValue(profile, null, indices, index);
		}
		
		Profile profile = (Profile)input;
		try {
			Object item = indices.get(index)[0]; //Currently, only use the first element of the index.
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
			LogUtil.trace(e);
		}
		
		return Constants.UNUSED;
	}


	/**
	 * Extract values regressors from input object.
	 * @param input specified input object which is often a profile.
	 * @param attList specified attribute list.
	 * @param indices specified list of indices.
	 * @return list of values of regressors from input object. Note that the list has form 1, x1, x2,..., xn in which the started value is always 1.
	 */
	public static double[] extractVariableValues(Object input, AttributeList attList, List<Object[]> indices) {
		if (input == null) return null;
		
		double[] xStatistic = new double[indices.size()];
		xStatistic[0] = 1;
		for (int j = 1; j < indices.size(); j++) {
			double xValue = extractVariableValue(input, attList, indices, j);
			if (Util.isUsed(xValue))
				xStatistic[j] = xValue;
			else
				xStatistic[j] = Constants.UNUSED;
		}
		
		return xStatistic;
	}

	
    /**
     * Creating graph for response variable given large statistic and regression model.
     * @param rm given regression model.
     * @param stats given large statistic.
     * @return graph for response variable.
     * @throws RemoteException if any error raises.
     */
    public static Graph createResponseGraph(RM rm, LargeStatistics stats) throws RemoteException {
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
				try {
					return "R=" + MathUtil.format(rm.calcR(1.0), 2);
				} catch (Exception e) {LogUtil.trace(e);}
				
				return "R=NaN";
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
     * @throws RemoteException if any error raises.
     */
    public static Graph createErrorGraph(RM rm, LargeStatistics stats) throws RemoteException {
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
     * @throws RemoteException if any error raises.
     */
    public static List<Graph> createResponseRalatedGraphs(RM rm) throws RemoteException {
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
     * @throws RemoteException if any error raises.
     */
	public static double calcVariance(RM rm, LargeStatistics stats) throws RemoteException {
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
     * @param factor multiplied factor.
     * @return correlation with specified regression model and large statistics.
     * @throws RemoteException if any error raises.
     */
	public static double calcR(RM rm, LargeStatistics stats, double factor) throws RemoteException {
		return calcR(rm, stats, factor, -1);
	}


    /**
     * Calculating correlation with specified regression model and large statistics.
     * @param rm specified regression model.
     * @param stats specified large statistics.
     * @param index if index < 0, calculating the correlation between estimated Z and real Z.
     * If index >= 0, calculating the correlation between real indexed X and real Z; note, X index from 1 because of X = (1, x1, x2,..., x(n-1)).
     * @param factor multiplied factor.
     * @return correlation with specified regression model and large statistics.
     * @throws RemoteException if any error raises.
     */
	public static double calcR(RM rm, LargeStatistics stats, double factor, int index) throws RemoteException {
		if (rm == null || stats == null) return Constants.UNUSED;
		
		Vector2 zVector = new Vector2(stats.size(), 0);
		Vector2 zEstimatedVector = new Vector2(stats.size(), 0);
		for (int i = 0; i < stats.size(); i++) {
            double z = index < 0 ? (double)rm.transformResponse(stats.getZData().get(i)[1], true) : stats.getXData().get(i)[index];
            zVector.set(i, z*factor);
            
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
     * @throws RemoteException if any error raises.
     */
	public static double[] calcError(RM rm, LargeStatistics stats) throws RemoteException {
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
	public static boolean saveLargeStatistics(RM rm, LargeStatistics stats, xURI uri, int decimal) {
		if (rm == null || stats == null || stats.size() == 0 || uri == null)
			return false;
		
		UriAssoc uriAssoc = Util.getFactory().createUriAssoc(uri);
		if (uriAssoc == null) return false;
		
		try {
			Writer writer = uriAssoc.getWriter(uri, false);
			
			StringBuffer columns = new StringBuffer();
			List<VarWrapper> regressors = rm.extractRegressors();
			for (int i = 0; i < regressors.size(); i++) {
				VarWrapper regressor = regressors.get(i);
				if (i > 0)
					columns.append(", ");
				columns.append(regressor.toString() + "~real");
			}
			VarWrapper response = rm.extractResponse();
			columns.append(", " + response.toString() + "~real");
			writer.write(columns.toString());
			
			int N = stats.size();
			for (int i = 0; i < N; i++) {
				double[] xVector = stats.getXData().get(i);
				double[] zVector = stats.getZData().get(i);
				StringBuffer row = new StringBuffer(xVector.length + 1);
				
				row.append("\n");
				for (int j = 1; j < xVector.length; j++) {
					if (decimal > 0)
						row.append(MathUtil.format(xVector[j], decimal));
					else
						row.append(xVector[j]);
						
					row.append(", ");
				}
				if (decimal > 0)
					row.append(MathUtil.format(zVector[1], decimal));
				else
					row.append(zVector[1]);
				
				writer.write(row.toString());
			}

			writer.close();
			return true;
		}
		catch (Exception e) {
			LogUtil.trace(e);
		}
		
		return false;
	}


	/**
	 * Generating 2-dimensional regressive data with specified list of regression coefficients, list of probabilities, list of variances, and size.
	 * Generated data for regressors x1, x2,..., xn is in [0, 1].
	 * Following is a code snippet to generate normal data for regression model:<br>
	 * <br>
	 * <code>
	 * List&lt;double[]&gt; alphas = Util.newList(2);<br>
	 * alphas.add(new double[] {0, 1});<br>
	 * alphas.add(new double[] {1, -1});<br>
	 * List&lt;Double&gt; probs = Util.newList(2);<br>
	 * probs.add(0.5);<br>
	 * probs.add(0.5);<br>
	 * List&lt;Double&gt; variances = Util.newList(2);<br>
	 * variances.add(0.001);<br>
	 * variances.add(0.001);<br>
	 * LargeStatistics stats = RMAbstract.generate2DRegressiveGaussianData2(alphas, probs, variances, 10000);
	 * </code>
	 * <br>
	 * @param alphas specified list of regression coefficients.
	 * @param probs specified list of probabilities. Each probability in this parameter specifies the frequency to fill in data for corresponding regression model.
	 * For instance, given 1000 times, if probs[1] = 0.4 and probs[2] = 0.6 then, the frequencies to generate data for the first model and second model are 400 and 600 times, respectively.
	 * @param variances specified list of variances.
	 * @param size size of data.
	 * @return regressive large statistics generated from specified list of regression coefficients, list of probabilities, list of variances, and size.
	 */
	public static LargeStatistics generate2DRegressiveGaussianData(List<double[]> alphas, List<Double> probs, List<Double> variances, int size) {
		if (alphas.size() == 0) return null;
		List<double[]> xData = Util.newList(size);
		List<double[]> zData = Util.newList(size);
		
		Random cRnd = new Random();
		Random xRnd = new Random();
		List<Random> zRnds = Util.newList(alphas.size());
		for (int k = 0; k < alphas.size(); k++) {
			zRnds.add(new Random());
		}

		//Indexing
		int[] counts = new int[alphas.size()];
		List<Integer> numbers = Util.newList(alphas.size());
		int m = 1000;
		for (int i = 0; i < alphas.size(); i++) {
			counts[i] = (int) (m * probs.get(i) + 0.5);
			if (probs.get(i) > 0) numbers.add(i);
		}
		List<Integer> indices = Util.newList(m);
		while (numbers.size() > 0) {
			int index = numbers.get(cRnd.nextInt(numbers.size()));
			if (counts[index] > 0)
				counts[index] = counts[index] - 1;
			if (counts[index] == 0) {
				Object o = Integer.valueOf(index);
				numbers.remove(o);
			}
			indices.add(index);
		}
		
		for (int i = 0; i < size; i++) {
			double[] xVector = new double[2];
			xVector[0] = 1;
			xData.add(xVector);
			double[] zVector = new double[2];
			zVector[0] = 1;
			zData.add(zVector);
			
			xVector[1] = xRnd.nextDouble();
			int index = indices.get(cRnd.nextInt(indices.size()));
			double[] alpha = alphas.get(index);
			double mean = alpha[0] + alpha[1] * xVector[1];
			zVector[1] = zRnds.get(index).nextGaussian() * Math.sqrt(variances.get(index)) + mean;
		}
		
		return new LargeStatistics(xData, zData);
	}
	
	
	/**
	 * Generating 2-dimensional regressive data with specified list of regression coefficients, list of probabilities, list of variances, and size.
	 * Generated data for regressors x1, x2,..., xn is in [0, 1].
	 * The difference between this method and {@link #generate2DRegressiveGaussianData(List, List, List, int)} is that this method generates data for x1, x2,..., xn according to intervals.
	 * For instance, if the number models is k = alphas.size(), the intervals to generate data of xi for k models are (0, 1/k), (1/k, 2/k),..., and ((k-1)/k, 1), respectively.
	 * Following is a code snippet to generate normal data for regression model:<br>
	 * <br>
	 * <code>
	 * List&lt;double[]&gt; alphas = Util.newList(2);<br>
	 * alphas.add(new double[] {0, 1});<br>
	 * alphas.add(new double[] {1, -1});<br>
	 * List&lt;Double&gt; probs = Util.newList(2);<br>
	 * probs.add(0.5);<br>
	 * probs.add(0.5);<br>
	 * List&lt;Double&gt; variances = Util.newList(2);<br>
	 * variances.add(0.001);<br>
	 * variances.add(0.001);<br>
	 * LargeStatistics stats = RMAbstract.generate2DRegressiveGaussianData2(alphas, probs, variances, 10000);
	 * </code>
	 * <br>
	 * @param alphas specified list of regression coefficients.
	 * @param probs specified list of probabilities. Each probability in this parameter specifies the frequency to fill in data for corresponding regression model.
	 * For instance, given 1000 times, if probs[1] = 0.4 and probs[2] = 0.6 then, the frequencies to generate data for the first model and second model are 400 and 600 times, respectively.
	 * @param variances specified list of variances.
	 * @param size size of data.
	 * @return regressive large statistics generated from specified list of regression coefficients, list of probabilities, list of variances, and size.
	 */
	public static LargeStatistics generate2DRegressiveGaussianDataWithXIntervals(List<double[]> alphas, List<Double> probs, List<Double> variances, int size) {
		if (alphas.size() == 0) return null;
		List<double[]> xData = Util.newList(size);
		List<double[]> zData = Util.newList(size);
		
		Random cRnd = new Random(); //for randomizing indices.
		Random xRnd = new Random(); //for randomizing x1, x2,..., xn
		List<Random> zRnds = Util.newList(alphas.size()); //for randomizing z
		for (int k = 0; k < alphas.size(); k++) {
			zRnds.add(new Random());
		}

		//Indexing which aims to filling regression models (x1, x2,..., xn) according to random order.
		int[] counts = new int[alphas.size()];
		List<Integer> numbers = Util.newList(alphas.size());
		int m = 1000;
		for (int i = 0; i < alphas.size(); i++) {
			counts[i] = (int) (m * probs.get(i) + 0.5);
			if (probs.get(i) > 0) numbers.add(i);
		}
		List<Integer> indices = Util.newList(m);
		while (numbers.size() > 0) {
			int index = numbers.get(cRnd.nextInt(numbers.size()));
			if (counts[index] > 0)
				counts[index] = counts[index] - 1;
			if (counts[index] == 0) {
				Object o = Integer.valueOf(index);
				numbers.remove(o);
			}
			indices.add(index);
		}
		
		//Interval
		List<double[]> intervals= Util.newList(alphas.size()); //Each interval specifies range of a regressor (Xi).  
		for (int i = 0 ; i < alphas.size(); i++) {
			double d = 1.0 / alphas.size();
			double a = i * d;
			double b = (i + 1) * d;
			intervals.add(new double[] {a, b});
		}
		
		for (int i = 0; i < size; i++) {
			double[] xVector = new double[2];
			xVector[0] = 1;
			xData.add(xVector);
			double[] zVector = new double[2];
			zVector[0] = 1;
			zData.add(zVector);
			
			int index = indices.get(cRnd.nextInt(indices.size()));
			double a = intervals.get(index)[0];
			double b = intervals.get(index)[1];
			xVector[1] = xRnd.nextDouble() * (b - a) + a;
			
			double[] alpha = alphas.get(index);
			double mean = alpha[0] + alpha[1] * xVector[1];
			zVector[1] = zRnds.get(index).nextGaussian() * Math.sqrt(variances.get(index)) + mean;
		}
		
		return new LargeStatistics(xData, zData);
	}


}
