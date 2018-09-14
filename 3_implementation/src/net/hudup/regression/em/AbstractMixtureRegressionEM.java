package net.hudup.regression.em;

import static net.hudup.regression.AbstractRegression.extractNumber;
import static net.hudup.regression.AbstractRegression.splitIndices;
import static net.hudup.regression.em.RegressionEMImpl.R_CALC_VARIANCE_FIELD;

import java.util.List;

import net.hudup.core.Util;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Dataset;
import net.hudup.core.data.Fetcher;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.MathUtil;
import net.hudup.em.ExponentialEM;
import net.hudup.regression.Regression;

/**
 * This abstract class implements partially expectation maximization (EM) algorithm for mixture regression models.
 * All algorithms that implement mixture regression model should derive from this class.
 *  
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class AbstractMixtureRegressionEM extends ExponentialEM implements Regression {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * List of internal regression model as parameter.
	 */
	protected List<RegressionEMImpl> rems = null;

	
	/**
	 * Setting up this model from other model.
	 * @param other other model.
	 * @throws Exception if any error raises.
	 */
	public void setup(DefaultMixtureRegressionEM other) throws Exception {
		// TODO Auto-generated method stub
		super.setup((Dataset)null, other);
	}

	
	@Override
	public Object learn(Object...info) throws Exception {
		// TODO Auto-generated method stub
		boolean prepared = false;
		if (info == null || info.length == 0 || !(info[0] instanceof DefaultMixtureRegressionEM))
			prepared = prepareInternalData(this.sample);
		else
			prepared = prepareInternalData((AbstractMixtureRegressionEM)info[0]);
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
			for (RegressionEMImpl rem : this.rems)
				rem.unsetup();
		}
	}

	
	/**
	 * Preparing data with other regression mixture model.
	 * @param other other regression mixture model.
	 * @return true if data preparation is successful.
	 * @throws Exception if any error raises.
	 */
	protected boolean prepareInternalData(AbstractMixtureRegressionEM other) throws Exception {
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
			RegressionEMImpl rem = createRegressionEM();
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
	protected RegressionEMImpl createRegressionEM() {
		RegressionEMImpl rem = new RegressionEMImpl() {

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

		};
		rem.getConfig().put(R_CALC_VARIANCE_FIELD, true);
		
		return rem;
	}
	
	
	/**
	 * Clear all internal data.
	 */
	protected void clearInternalData() {
		this.currentIteration = 0;
		this.estimatedParameter = this.currentParameter = this.previousParameter = null;
		
		if (this.rems != null) {
			for (RegressionEMImpl rem : this.rems) {
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
			RegressionEMImpl rem = this.rems.get(k);
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
			RegressionEMImpl rem = this.rems.get(k);
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
			RegressionEMImpl rem = this.rems.get(k);
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
			RegressionEMImpl rem = this.rems.get(k);
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
			RegressionEMImpl rem = this.rems.get(k);
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
			RegressionEMImpl rem = this.rems.get(k);
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
	public synchronized Object execute(Object input) {
		// TODO Auto-generated method stub
		if (this.rems == null || this.rems.size() == 0)
			return null;
		
		double result = 0;
		for (RegressionEMImpl rem : this.rems) {
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
			Regression regression = this.rems.get(i);
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
			
		for (RegressionEMImpl rem : this.rems) {
			if (rem == null) continue;
			double value = extractNumber(rem.extractResponse(input));
			if (Util.isUsed(value))
				return value;
		}
		return null;
	}


}
