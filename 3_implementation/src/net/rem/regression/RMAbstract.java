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
import java.util.List;
import java.util.Random;

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
import net.rem.regression.logistic.speqmath.Parser;
import net.rem.regression.ui.RMInspector;
import net.rem.regression.ui.graph.Graph;
import net.rem.regression.ui.graph.PlotGraphExt;
import net.rem.regression.ui.graph.PlotGraphExt2;

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
		Indices.Used usedIndices = Indices.extractUsedIndices(info);
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
	 */
	@SuppressWarnings("unchecked")
	protected boolean prepareInternalData(int[] xIndicesUsed, int[] zIndicesUsed) {
		clearInternalData();
		
		//Begin parsing indices
		Indices indices = Indices.parse(getConfig().getAsString(RM_INDICES_FIELD), (Fetcher<Profile>)sample, xIndicesUsed, zIndicesUsed);
		if (indices == null) return false;
		this.attList = indices.attList;
		this.xIndices = indices.xIndices;
		this.zIndices = indices.zIndices;
		//End parsing indices

		return true;
	}

	
	/**
	 * Preparing data.
	 * @return true if data preparation is successful.
	 * @throws RemoteException if any error raises.
	 */
	protected boolean prepareInternalData() {
		return prepareInternalData(null, null);
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
				return new RMInspector(UIUtil.getDialogForComponent(null), rm);
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
	public AttributeList getAttributeList() throws RemoteException {
		return attList;
	}


	@Override
	public Object transformRegressor(Object x, boolean inverse) throws RemoteException {
		return x;
	}


	@Override
	public Object transformResponse(Object z, boolean inverse) throws RemoteException {
		return z;
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
		if (input == null)
			return Constants.UNUSED;
		else if (input instanceof Profile)
			return extractVariableValue(input, null, zIndices, 1);
		else
			return extractVariableValue(input, attList, zIndices, 1);
	}


	@Override
	public DataConfig createDefaultConfig() {
		DataConfig config = super.createDefaultConfig();
		config.put(RM_INDICES_FIELD, RM_INDICES_DEFAULT);
		return config;
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

//    	PlotGraphExt pg = new PlotGraphExt(data) { //Removed due to remote casting.
//			/**
//			 * Serial version UID for serializable class.
//			 */
//			private static final long serialVersionUID = 1L;
//
//			@Override
//			public String getGraphFeature() {
//				try {
//					return "R=" + MathUtil.format(rm.calcR(1.0), 2);
//				} catch (Exception e) {LogUtil.trace(e);}
//				
//				return "R=NaN";
//			}
//    	};
    	String graphFeature = "R=NaN";
		try {
			graphFeature = "R=" + MathUtil.format(rm.calcR(), 2);
		} catch (Exception e) {LogUtil.trace(e);}
		PlotGraphExt pg = new PlotGraphExt2(data, graphFeature);
    	
    	
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

    	double mean = errorMean, sd = errorSd;
//    	PlotGraphExt pg = new PlotGraphExt(data) { //Removed due to remote casting.
//
//			/**
//			 * Serial version UID for serializable class.
//			 */
//			private static final long serialVersionUID = 1L;
//
//			@Override
//			public String getGraphFeature() {
//				return MathUtil.format(mean, 2) + " +/- 1.96*" + MathUtil.format(sd, 2);
//			}
//    	};
    	String graphFeature = MathUtil.format(mean, 2) + " +/- 1.96*" + MathUtil.format(sd, 2);
		PlotGraphExt pg = new PlotGraphExt2(data, graphFeature);

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
     * Calculating correlation between real response and estimated response.
     * @param rm specified regression model.
     * @param stats specified large statistics.
     * @return correlation between real response and estimated response.
     * @throws RemoteException if any error raises.
     */
	public static double calcR(RM rm, LargeStatistics stats) throws RemoteException {
		return calcR(rm, stats, -1);
	}


    /**
     * Calculating correlation with real response or real regressor and estimated response.
     * @param rm specified regression model.
     * @param stats specified large statistics.
     * @param index if index < 0, calculating the correlation between estimated Z and real Z.
     * If index >= 0, calculating the correlation between real indexed X and real Z; note, X index from 1 because of X = (1, x1, x2,..., x(n-1)).
     * @return correlation with real response or real regressor and estimated response.
     * @throws RemoteException if any error raises.
     */
	public static double calcR(RM rm, LargeStatistics stats, int index) throws RemoteException {
		if (rm == null || stats == null) return Constants.UNUSED;
		
		Vector2 zVector = new Vector2(stats.size(), 0);
		Vector2 zEstimatedVector = new Vector2(stats.size(), 0);
		for (int i = 0; i < stats.size(); i++) {
            double z = index < 0 ? (double)rm.transformResponse(stats.getZData().get(i)[1], true) : stats.getXData().get(i)[index];
            zVector.set(i, z);
            
            double zEstimated = rm.executeByXStatistic(stats.getXData().get(i));
            zEstimatedVector.set(i, zEstimated);
		}
		
		return zEstimatedVector.corr(zVector);
	}

	
	/**
	 * Calculating the correlation between real regressor and real response.
	 * @param stats specified large statistics.
	 * @param regressorIndex repressor index from 1 because of X = (1, x1, x2,..., x(n-1)).
	 * @return correlation between real regressor and real response.
	 */
	public static double calcRRegressorResponse(LargeStatistics stats, int regressorIndex) {
		if (stats == null || regressorIndex < 0) return Constants.UNUSED;
		
		Vector2 zVector = new Vector2(stats.size(), 0);
		Vector2 xVector = new Vector2(stats.size(), 0);
		for (int i = 0; i < stats.size(); i++) {
            double z = stats.getZData().get(i)[1];
            zVector.set(i, z);
            
            double x = stats.getXData().get(i)[regressorIndex];
            xVector.set(i, x);
		}
		
		return zVector.corr(xVector);

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
