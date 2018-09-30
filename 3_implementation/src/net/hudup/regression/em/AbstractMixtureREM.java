package net.hudup.regression.em;

import static net.hudup.regression.AbstractRM.extractNumber;
import static net.hudup.regression.AbstractRM.splitIndices;
import static net.hudup.regression.em.REMImpl.R_CALC_VARIANCE_FIELD;

import java.awt.Color;
import java.util.List;

import javax.swing.JOptionPane;

import flanagan.math.Fmath;
import flanagan.plot.PlotGraph;
import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Dataset;
import net.hudup.core.data.Fetcher;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.MathUtil;
import net.hudup.core.logistic.Vector2;
import net.hudup.core.logistic.ui.UIUtil;
import net.hudup.em.ExponentialEM;
import net.hudup.regression.AbstractRM.VarWrapper;
import net.hudup.regression.RM;
import net.hudup.regression.RM2;
import net.hudup.regression.em.ui.REMDlg;
import net.hudup.regression.em.ui.graph.Graph;
import net.hudup.regression.em.ui.graph.PlotGraphExt;

/**
 * This abstract class implements partially expectation maximization (EM) algorithm for mixture regression models.
 * All algorithms that implement mixture regression model should derive from this class.
 *  
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class AbstractMixtureREM extends ExponentialEM implements RM2 {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * List of internal regression model as parameter.
	 */
	protected List<REMImpl> rems = null;

	
	/**
	 * Setting up this model from other model.
	 * @param other other model.
	 * @throws Exception if any error raises.
	 */
	public void setup(DefaultMixtureREM other) throws Exception {
		// TODO Auto-generated method stub
		super.setup((Dataset)null, other);
	}

	
	@Override
	public Object learn(Object...info) throws Exception {
		// TODO Auto-generated method stub
		boolean prepared = false;
		if (info == null || info.length == 0 || !(info[0] instanceof DefaultMixtureREM))
			prepared = prepareInternalData(this.sample);
		else
			prepared = prepareInternalData((AbstractMixtureREM)info[0]);
		if (!prepared) {
			clearInternalData();
			return null;
		}
		
		if (super.learn(info) == null) {
			clearInternalData();
			return null;
		}
		
		if(!adjustMixtureParameters()) {
			clearInternalData();
			return null;
		}
		
		return this.rems;
	}
	
	
	@Override
	public synchronized void unsetup() {
		// TODO Auto-generated method stub
		super.unsetup();
		if (this.rems != null) {
			for (REMImpl rem : this.rems)
				rem.unsetup();
		}
	}

	
	/**
	 * Preparing data with other regression mixture model.
	 * @param other other regression mixture model.
	 * @return true if data preparation is successful.
	 * @throws Exception if any error raises.
	 */
	protected boolean prepareInternalData(AbstractMixtureREM other) throws Exception {
		return prepareInternalData(this.sample);
	}
	
	
	/**
	 * Preparing data.
	 * @param inputSample specified sample.
	 * @return true if data preparation is successful.
	 * @throws Exception if any error raises.
	 */
	protected boolean prepareInternalData(Fetcher<Profile> inputSample) throws Exception {
		clearInternalData();
		DataConfig thisConfig = this.getConfig();
		
		List<String> indicesList = splitIndices(thisConfig.getAsString(R_INDICES_FIELD));
		if (indicesList.size() == 0) {
			AttributeList attList = getSampleAttributeList(inputSample);
			if (attList.size() < 2)
				return false;
			
			StringBuffer indices = new StringBuffer();
			for (int i = 1; i <= attList.size(); i++) {// For fair test
				if (i > 1)
					indices.append(", ");
				indices.append(i);
			}
			indicesList.add(indices.toString());
		}
		
		this.rems = Util.newList(indicesList.size());
		for (int i = 0; i < indicesList.size(); i++) {
			REMImpl rem = createRegressionEM();
			rem.getConfig().put(R_INDICES_FIELD, indicesList.get(i));
			rem.setup(inputSample);
			if(rem.attList != null) // if rem is set up successfully.
				this.rems.add(rem);
		}
		
		if (this.rems.size() == 0) {
			this.rems = null;
			return false;
		}
		else
			return true;
	}
	
	
	/**
	 * Creating internal regression model.
	 * @return internal regression model.
	 */
	protected REMImpl createRegressionEM() {
		REMImpl rem = new REMImpl() {

			/**
			 * Serial version UID for serializable class.
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public synchronized Object learn(Object...info) throws Exception {
				// TODO Auto-generated method stub
				boolean prepared = prepareInternalData(sample);
				if (prepared)
					return prepared;
				else
					return null;
			}

			@Override
			protected Object transformRegressor(Object x, boolean inverse) {
				// TODO Auto-generated method stub
				return getMixtureREM().transformRegressor(x, inverse);
			}

			@Override
			protected Object transformResponse(Object z, boolean inverse) {
				// TODO Auto-generated method stub
				return getMixtureREM().transformResponse(z, inverse);
			}
			
		};
		rem.getConfig().put(R_CALC_VARIANCE_FIELD, true);
		
		return rem;
	}
	
	
	/**
	 * Getting this mixture regression expectation maximization model.
	 * @return this mixture regression expectation maximization model.
	 */
	protected AbstractMixtureREM getMixtureREM() {
		return this;
	}
	
	
	/**
	 * Clear all internal data.
	 */
	protected void clearInternalData() {
		this.currentIteration = 0;
		this.estimatedParameter = this.currentParameter = this.previousParameter = null;
		
		if (this.rems != null) {
			for (REMImpl rem : this.rems) {
				rem.clearInternalData();
				rem.unsetup();
			}
			this.rems.clear();
			this.rems = null;
		}
		
		this.statistics = null;
	}

	
	/**
	 * Expectation method of this class does not change internal data.
	 */
	@Override
	protected Object expectation(Object currentParameter, Object...info) throws Exception {
		// TODO Auto-generated method stub
		if (currentParameter == null)
			return null;
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> parameters = (List<ExchangedParameter>)currentParameter;
		List<LargeStatistics> stats = Util.newList(this.rems.size());
		for (int k = 0; k < this.rems.size(); k++) {
			REMImpl rem = this.rems.get(k);
			ExchangedParameter parameter = parameters.get(k);
//			if (rem.terminatedCondition(rem.getEstimatedParameter(), rem.getCurrentParameter(), rem.getPreviousParameter(), info)
//					&& rem.getLargeStatistics() != null)
//				continue;
				
			LargeStatistics stat = (LargeStatistics)rem.expectation(parameter);
			rem.setStatistics(stat);
			stats.add(stat);
			
//			if (stat == null)
//				logger.error("Some regression models are failed in expectation");
		}
		
		return stats;
	}

	
	/**
	 * Maximization method of this class changes internal data.
	 */
	@Override
	protected Object maximization(Object currentStatistic, Object...info) throws Exception {
		// TODO Auto-generated method stub
		if (currentStatistic == null)
			return null;
		@SuppressWarnings("unchecked")
		List<LargeStatistics> stats = (List<LargeStatistics>)currentStatistic;
		List<ExchangedParameter> parameters = Util.newList(this.rems.size());
		for (int k = 0; k < this.rems.size(); k++) {
			REMImpl rem = this.rems.get(k);
			LargeStatistics stat = stats.get(k);
//			if (rem.terminatedCondition(rem.getEstimatedParameter(), rem.getCurrentParameter(), rem.getPreviousParameter(), info)
//					&& rem.getLargeStatistics() != null)
//				continue;

			ExchangedParameter parameter = (ExchangedParameter)rem.maximization(stat);
			rem.setEstimatedParameter(parameter);
			parameters.add(parameter);
			
//			if (parameter == null)
//				logger.error("Some regression models are failed in expectation");
		}
		
		return parameters;
	}

	
	/**
	 * Initialization method of this class changes internal data.
	 */
	@Override
	protected Object initializeParameter() {
		// TODO Auto-generated method stub
		List<ExchangedParameter> parameters = Util.newList(this.rems.size());
		for (int k = 0; k < this.rems.size(); k++) {
			REMImpl rem = this.rems.get(k);
			ExchangedParameter parameter = (ExchangedParameter)rem.initializeParameter();
			
			rem.setEstimatedParameter(parameter);
			rem.setCurrentParameter(parameter);
			rem.setPreviousParameter(null);
			rem.setStatistics(null);
			rem.setCurrentIteration(this.getCurrentIteration());

			parameters.add(parameter);
		}
		
		for (ExchangedParameter parameter : parameters) {
			parameter.setCoeff(1.0 / (double)this.rems.size());
			if (!Util.isUsed(parameter.getZVariance()))
				parameter.setZVariance(1.0);
		}
		return parameters;
	}

	
	@Override
	protected void permuteNotify() {
		// TODO Auto-generated method stub
		super.permuteNotify();
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> estimatedParameters = (List<ExchangedParameter>)getEstimatedParameter();
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> currentParameters = (List<ExchangedParameter>)getCurrentParameter();
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> previousParameters = (List<ExchangedParameter>)getPreviousParameter();
		
		for (int k = 0; k < this.rems.size(); k++) {
			REMImpl rem = this.rems.get(k);
			ExchangedParameter estimatedParameter = estimatedParameters.get(k);
			ExchangedParameter currentParameter = currentParameters.get(k);
			ExchangedParameter previousParameter = previousParameters != null ? previousParameters.get(k) : null;
			
			rem.setEstimatedParameter(estimatedParameter);
			rem.setCurrentParameter(currentParameter);
			rem.setPreviousParameter(previousParameter);
			rem.setCurrentIteration(this.getCurrentIteration());
		}
	}


	@Override
	protected void finishNotify() {
		// TODO Auto-generated method stub
		super.finishNotify();
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> estimatedParameters = (List<ExchangedParameter>)getEstimatedParameter();
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> currentParameters = (List<ExchangedParameter>)getCurrentParameter();
		
		for (int k = 0; k < this.rems.size(); k++) {
			REMImpl rem = this.rems.get(k);
			ExchangedParameter estimatedParameter = estimatedParameters.get(k);
			ExchangedParameter currentParameter = currentParameters.get(k);
			
			rem.setEstimatedParameter(estimatedParameter);
			rem.setCurrentParameter(currentParameter);
			rem.setCurrentIteration(this.getCurrentIteration());
		}
	}


	/**
	 * Adjusting specified parameters based on specified statistics according to mixture model in one iteration.
	 * This method does not need a loop because both mean and variance were optimized in REM process and so the probabilities of components will be optimized in only one time.
	 * @return true if the adjustment process is successful.
	 * @throws Exception if any error raises.
	 */
	protected boolean adjustMixtureParameters() throws Exception {
		// TODO Auto-generated method stub
		//Do nothing
		return true;
	}


	@Override
	protected boolean terminatedCondition(Object estimatedParameter, Object currentParameter, Object previousParameter, Object... info) {
		// TODO Auto-generated method stub
		if (this.rems == null)
			return true;

		@SuppressWarnings("unchecked")
		List<ExchangedParameter> estimatedParameters = (List<ExchangedParameter>)estimatedParameter;
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> currentParameters = (List<ExchangedParameter>)currentParameter;
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> previousParameters = (List<ExchangedParameter>)previousParameter;
		
		boolean terminated = true;
		for (int k = 0; k < this.rems.size(); k++) {
			REMImpl rem = this.rems.get(k);
			ExchangedParameter pParameter = previousParameters != null ? previousParameters.get(k) : null;
			ExchangedParameter cParameter = currentParameters.get(k);
			ExchangedParameter eParameter = estimatedParameters.get(k);
			
			if (eParameter == null && cParameter == null)
				continue;
			else if (eParameter == null || cParameter == null)
				return false;
			
			terminated = terminated && rem.terminatedCondition(eParameter, cParameter, pParameter, info);
			if (!terminated)
				return false;
		}
		
		return terminated;
	}

	
	@Override
	public synchronized Object executeByXStatistic(double[] xStatistic) {
		if (this.rems == null || this.rems.size() == 0 || xStatistic == null)
			return null;
		
		double result = 0;
		for (REMImpl rem : this.rems) {
			ExchangedParameter parameter = (ExchangedParameter)rem.getParameter();
			
			double value = extractNumber(rem.executeByXStatistic(xStatistic));
			if (Util.isUsed(value))
				result += parameter.getCoeff() * value;
			else
				return null;
		}
		return result;
	}
	
	
	@Override
	public synchronized Object execute(Object input) {
		// TODO Auto-generated method stub
		if (this.rems == null || this.rems.size() == 0)
			return null;
		
		double result = 0;
		for (REMImpl rem : this.rems) {
			ExchangedParameter parameter = (ExchangedParameter)rem.getParameter();
			
			double value = extractNumber(rem.execute(input));
			if (Util.isUsed(value))
				result += parameter.getCoeff() * value;
			else
				return null;
		}
		return result;
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
	public synchronized String parameterToShownText(Object parameter, Object... info) {
		// TODO Auto-generated method stub
		if (parameter == null || !(parameter instanceof List<?>))
			return "";
		
		@SuppressWarnings("unchecked")
		List<ExchangedParameter> parameters = (List<ExchangedParameter>)parameter;
		StringBuffer buffer = new StringBuffer();
		for (int k = 0; k < rems.size(); k++) {
			if (k > 0)
				buffer.append(", ");
			String text = rems.get(k).parameterToShownText(parameters.get(k), info);
			buffer.append("{" + text + "}");
		}
		buffer.append(": ");
		buffer.append("t=" + MathUtil.format(getCurrentIteration()));
		
		return buffer.toString();
	}

	
	@Override
	public synchronized String getDescription() {
		// TODO Auto-generated method stub
		if (this.rems == null)
			return "";

		StringBuffer buffer = new StringBuffer();
		for (int i = 0; i < this.rems.size(); i++) {
			RM regression = this.rems.get(i);
			if (i > 0)
				buffer.append(", ");
			String text = "";
			if (regression != null)
				text = regression.getDescription();
			buffer.append("{" + text + "}");
		}
		buffer.append(": ");
		buffer.append("t=" + getCurrentIteration());
		
		return buffer.toString();
	}

	
	@Override
	public synchronized void manifest() {
		// TODO Auto-generated method stub
		if (getParameter() == null) {
			JOptionPane.showMessageDialog(
					UIUtil.getFrameForComponent(null), 
					"Invalid regression model", 
					"Invalid regression model", 
					JOptionPane.ERROR_MESSAGE);
		}
		else
			new REMDlg(UIUtil.getFrameForComponent(null), this);
	}


	@Override
	public DataConfig createDefaultConfig() {
		// TODO Auto-generated method stub
		DataConfig config = super.createDefaultConfig();
		config.put(R_INDICES_FIELD, R_INDICES_DEFAULT);
		return config;
	}

	
	@Override
	public synchronized Object extractResponse(Object input) {
		// TODO Auto-generated method stub
		if (this.rems == null)
			return null;
			
		for (REMImpl rem : this.rems) {
			if (rem == null) continue;
			double value = extractNumber(rem.extractResponse(input));
			if (Util.isUsed(value))
				return value;
		}
		return null;
	}


	@Override
	public String extractResponseName() {
		// TODO Auto-generated method stub
		if (this.rems == null || this.rems.size() == 0)
			return null;
		else
			return this.rems.get(0).extractResponseName(); // Suppose all REMs have the same response variable;
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
	
	
	@Override
	public synchronized Graph createRegressorGraph(int xIndex) {
		// TODO Auto-generated method stub
		if (this.rems == null || this.rems.size() == 0)
			return null;

		LargeStatistics stats = this.rems.get(0).getLargeStatistics(); // Suppose all REM has the same large statistics.
    	int ncurves = 1 + this.rems.size();
    	int npoints = stats.size();
    	double[][] data = PlotGraph.data(ncurves, npoints);
    	int[] popt = new int[1 + this.rems.size()];
    	int[] lopt = new int[1 + this.rems.size()];
    	
    	for(int i = 0; i < npoints; i++) {
            data[0][i] = stats.getXData().get(i)[xIndex];
            data[1][i] = stats.getZData().get(i)[1];
        }
    	popt[0] = 1;
    	lopt[0] = 0;
    	
    	for (int k = 0; k < this.rems.size(); k++) {
    		ExchangedParameter parameter = this.rems.get(k).getExchangedParameter();
    		double coeff0 = parameter.getAlpha().get(0);
    		double coeff1 = parameter.getAlpha().get(xIndex);
    		
        	data[2*(k+1)][0] = Fmath.minimum(data[0]);
        	data[2*(k+1) + 1][0] = coeff0 + coeff1 * data[2*(k+1)][0];
        	
        	data[2*(k+1)][1] = Fmath.maximum(data[0]);
        	data[2*(k+1) + 1][1] = coeff0 + coeff1 * data[2*(k+1)][1];
        	
        	popt[k + 1] = 0;
        	lopt[k + 1] = 3;
    	}

    	PlotGraphExt pg = new PlotGraphExt(data);

    	pg.setGraphTitle("Regressor plot");
    	pg.setXaxisLegend(this.rems.get(0).extractRegressorName(xIndex));
    	pg.setYaxisLegend(this.rems.get(0).extractResponseName());
    	pg.setPoint(popt);
    	pg.setLine(lopt);

    	pg.setBackground(Color.WHITE);
        return pg;
	}


	@Override
	public synchronized Graph createResponseGraph() {
		// TODO Auto-generated method stub
		if (this.rems == null || this.rems.size() == 0)
			return null;
		else
			return REMImpl.createResponseGraph(this, this.rems.get(0).getLargeStatistics()); // Suppose all REM has the same large statistics.
	}


	@Override
	public Graph createErrorGraph() {
		// TODO Auto-generated method stub
		if (this.rems == null || this.rems.size() == 0)
			return null;
		else
			return REMImpl.createErrorGraph(this, this.rems.get(0).getLargeStatistics()); // Suppose all REM has the same large statistics.
	}


	@Override
	public List<Graph> createResponseRalatedGraphs() {
		// TODO Auto-generated method stub
		if (this.rems == null || this.rems.size() == 0)
			return null;
		else
			return REMImpl.createResponseRalatedGraphs(this);
	}


	@Override
	public List<VarWrapper> getRegressorExpressions() {
		// TODO Auto-generated method stub
		List<VarWrapper> wholeVarList = Util.newList();
		if (this.rems == null || this.rems.size() == 0)
			return wholeVarList;
		
		for (REMImpl rem : this.rems) {
			List<VarWrapper> varList = rem.getRegressorExpressions();
			for (VarWrapper var : varList) {
				int found = VarWrapper.lookup(wholeVarList, var.getExpr(), false);
				if (found < 0)
					wholeVarList.add(var);
			}
		}
		return wholeVarList;
	}


	@Override
	public List<VarWrapper> getRegressors() {
		// TODO Auto-generated method stub
		List<VarWrapper> wholeVarList = Util.newList();
		if (this.rems == null || this.rems.size() == 0)
			return wholeVarList;
		
		for (REMImpl rem : this.rems) {
			List<VarWrapper> varList = rem.getRegressorExpressions();
			for (VarWrapper var : varList) {
				int found = VarWrapper.lookup(wholeVarList, var.getName(), true);
				if (found >= 0)
					wholeVarList.add(var);
			}
		}
		return wholeVarList;
	}


	@Override
	public synchronized double calcVariance() {
		// TODO Auto-generated method stub
		if (this.rems == null || this.rems.size() == 0)
			return Constants.UNUSED;
		
		LargeStatistics stats = this.rems.get(0).getLargeStatistics(); // Suppose all REM has the same large statistics.
		List<double[]> xData = stats.getXData();
		List<double[]> zData = stats.getZData();
		
		double ss = 0;
		int N = 0;
		for (int i = 0; i < xData.size(); i++) {
			double[] xVector = xData.get(i);
			double zValue = zData.get(i)[1];
			double zEstimatedValue = (double)this.executeByXStatistic(xVector);
			
			if (Util.isUsed(zValue) && Util.isUsed(zEstimatedValue)) {
				ss += (zEstimatedValue - zValue) * (zEstimatedValue - zValue);
				N++;
			}
		}
		return ss / N;
	}


	@Override
	public double calcR() {
		// TODO Auto-generated method stub
		if (this.rems == null || this.rems.size() == 0)
			return Constants.UNUSED;
		
		LargeStatistics stats = this.rems.get(0).getLargeStatistics(); // Suppose all REM has the same large statistics.
		Vector2 zVector = new Vector2(stats.size(), 0);
		Vector2 zCalcVector = new Vector2(stats.size(), 0);
		for (int i = 0; i < stats.size(); i++) {
            double z = stats.getZData().get(i)[1];
            zVector.set(i, z);
            
            double zCalc = (double)executeByXStatistic(stats.getXData().get(i));
            zCalcVector.set(i, zCalc);
		}
		
		return zCalcVector.corr(zVector);
	}


	@Override
	public double[] calcError() {
		// TODO Auto-generated method stub
		if (this.rems == null || this.rems.size() == 0)
			return null;
		
		LargeStatistics stats = this.rems.get(0).getLargeStatistics(); // Suppose all REM has the same large statistics.
		Vector2 error = new Vector2(stats.size(), 0);
		for (int i = 0; i < stats.size(); i++) {
            double z = stats.getZData().get(i)[1];
            double zCalc = (double)executeByXStatistic(stats.getXData().get(i));
            error.set(i, zCalc - z);
		}
		
    	return new double[] {error.mean(), error.mleVar()};
	}
	
	
}
