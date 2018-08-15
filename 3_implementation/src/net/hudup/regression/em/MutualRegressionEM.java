package net.hudup.regression.em;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.Alg;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Profile;
import net.hudup.em.ExponentialEM;
import net.hudup.regression.AbstractRegression;
import net.hudup.regression.Regression;
import net.hudup.regression.em.RegressionEMImpl.ExchangedParameter;

/**
 * This class implements expectation maximization (EM) algorithm for many partial regression models.
 * If there are 2 partial models (by default), the mutual regression model becomes dual regression model.
 *  
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class MutualRegressionEM extends ExponentialEM implements Regression, DuplicatableAlg {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Name of model number field.
	 */
	protected final static String MODELS_NUMBER_FIELD = "mrem_models_number";
	
	
	/**
	 * Default number of models. The default number is 2, which implies dual regression model.
	 */
	protected final static int MODELS_NUMBER_DEFAULT = 2;

	
	/**
	 * List of internal regression model as parameter.
	 */
	protected List<RegressionEMImpl> rems = new ArrayList<>();

	
	@Override
	public Object learn() throws Exception {
		// TODO Auto-generated method stub
		if (!prepareInternalData()) {
			clearInternalData();
			return null;
		}
		
		ExchangedParameter[] parameters = (ExchangedParameter[])super.learn();
		if (parameters == null) {
			clearInternalData();
			return null;
		}
		
		boolean success = false;
		for (int i = 0; i < parameters.length; i++) {
			ExchangedParameter parameter = parameters[i];
			RegressionEMImpl rem = this.rems.get(i);
			if (parameter != null && rem != null) {
				rem.setParameter(parameter, this.getCurrentIteration());
				success = true;
			}
		}
		if (!success) {
			clearInternalData();
			return null;
		}
		else
			return parameters;
	}

	
	@Override
	public synchronized void unsetup() {
		// TODO Auto-generated method stub
		super.unsetup();
		for (RegressionEMImpl rem : this.rems) {
			if (rem != null)
				rem.unsetup();
		}
	}

	
	/**
	 * Preparing data.
	 * @return true if data preparation is successful.
	 * @throws Exception if any error raises.
	 */
	protected boolean prepareInternalData() throws Exception {
		clearInternalData();
		DataConfig thisConfig = this.getConfig();
		int modelNumbers = thisConfig.getAsInt(MODELS_NUMBER_FIELD);
		if (modelNumbers <= 0)
			return false;
		
		List<String> indices = AbstractRegression.splitIndices(thisConfig.getAsString(R_INDICES_FIELD));
		for (int i = 0; i < modelNumbers; i++) {
			RegressionEMImpl rem = new RegressionEMImpl() {

				/**
				 * Serial version UID for serializable class.
				 */
				private static final long serialVersionUID = 1L;

				@Override
				public synchronized Object learn() throws Exception {
					// TODO Auto-generated method stub
					return prepareInternalData();
				}

			};
			
			DataConfig config = rem.getConfig();
			if (i < indices.size())
				config.put(R_INDICES_FIELD, indices.get(i));
			config.put(RegressionEMImpl.REM_INVERSE_MODE_FIELD, thisConfig.get(RegressionEMImpl.REM_INVERSE_MODE_FIELD));
			config.put(RegressionEMImpl.REM_BALANCE_MODE_FIELD, thisConfig.get(RegressionEMImpl.REM_BALANCE_MODE_FIELD));
			rem.setup(this.sample);
			if(rem.attList != null) // if rem1 is set up successfully.
				this.rems.add(rem);
		}
		
		return (this.rems.size() == 0 ? false : true);
	}
	
	
	/**
	 * Clear all internal data.
	 */
	protected void clearInternalData() {
		this.currentIteration = 0;
		this.currentParameter = this.estimatedParameter = null;
		for (RegressionEMImpl rem : this.rems) {
			if (rem != null)
				rem.clearInternalData();
		}
		this.rems.clear();
	}

	
	@Override
	protected Object expectation(Object currentParameter) throws Exception {
		// TODO Auto-generated method stub
		ExchangedParameter[] exParameters = (ExchangedParameter[])currentParameter;
		
		ExchangedParameter[] stats = new ExchangedParameter[exParameters.length];
		Arrays.fill(stats, null);
		List<ExchangedParameter> successStats = new ArrayList<>();
		for (int i = 0; i < exParameters.length; i++) {
			ExchangedParameter exParameter = exParameters[i];
			RegressionEMImpl rem = this.rems.get(i);
			if (exParameter == null || rem == null)
				continue;
			
			stats[i] = (ExchangedParameter)rem.expectation(exParameter);
			if (stats[i] != null)
				successStats.add(stats[i]);
		}
		if (successStats.size() == 0)
			return null;
		
		//Retrieving same-length Z statistics
		int N = successStats.get(0).vector.length;
		List<ExchangedParameter> successStats2 = new ArrayList<>();
		for (ExchangedParameter stat : successStats) {
			if (stat.vector.length == N)
				successStats2.add(stat);
		}
		if (successStats2.size() <= 1)
			return stats;
		
		double[] meanVector = new double[N]; //Z statistic
		for (int j = 0; j < N; j++) {
			double mean = 0;
			for (ExchangedParameter stat : successStats2) {
				mean += stat.vector[j];
			}
			mean = mean / (double)successStats2.size();
			meanVector[j] = mean;
		}
		for (ExchangedParameter stat : successStats2) {
			stat.vector = meanVector;
		}
		
		return stats;
	}

	
	@Override
	protected Object maximization(Object currentStatistic) throws Exception {
		// TODO Auto-generated method stub
		ExchangedParameter[] stats = (ExchangedParameter[])currentStatistic;

		ExchangedParameter[] exParameters = new ExchangedParameter[stats.length];
		Arrays.fill(exParameters, null);
		boolean success = false;
		for (int i = 0; i < stats.length; i++) {
			ExchangedParameter stat = stats[i];
			RegressionEMImpl rem = this.rems.get(i);
			if (stat == null || rem == null)
				continue;
			
			exParameters[i] = (ExchangedParameter)rem.maximization(stat);
			if (exParameters[i] != null)
				success = true;
		}
		if (!success)
			return null;
		else
			return exParameters;
	}

	
	@Override
	protected Object initializeParameter() {
		// TODO Auto-generated method stub
		ExchangedParameter[] exParameters = new ExchangedParameter[this.rems.size()];
		Arrays.fill(exParameters, null);
		boolean success = false;
		for (int i = 0; i < this.rems.size(); i++) {
			RegressionEMImpl rem = this.rems.get(i);
			if (rem == null)
				continue;
			
			exParameters[i] = (ExchangedParameter)rem.initializeParameter();
			if (exParameters[i] != null)
				success = true;
		}
		if (!success)
			return null;
		else
			return exParameters;
	}

	
	@Override
	protected boolean terminatedCondition(Object currentParameter, Object estimatedParameter, Object... info) {
		// TODO Auto-generated method stub
		ExchangedParameter[] exCurrentParameters = (ExchangedParameter[])currentParameter;
		ExchangedParameter[] exEstimatedParameters = (ExchangedParameter[])estimatedParameter;

		boolean terminated = true;
		for (int i = 0; i < this.rems.size(); i++) {
			RegressionEMImpl rem = this.rems.get(i);
			ExchangedParameter exCurrentParameter = exCurrentParameters[i];
			ExchangedParameter exEstimatedParameter = exEstimatedParameters[i];
			
			if (rem == null || exCurrentParameter == null || exEstimatedParameter == null)
				continue;
			
			terminated = terminated && rem.terminatedCondition(exCurrentParameter, exEstimatedParameter, info);
			if (!terminated)
				return false;
		}
		
		return terminated;
	}

	
	@Override
	public synchronized Object execute(Object input) {
		// TODO Auto-generated method stub
		int k = 0;
		double mean = 0;
		for (RegressionEMImpl rem : this.rems) {
			if (rem == null)
				continue;
			
			Object value = rem.execute(input);
			if (value == null || !(value instanceof Number))
				continue;
			
			double realValue = ((Number)value).doubleValue();
			if (Util.isUsed(realValue)) {
				mean += realValue;
				k++;
			}
		}
		if (k == 0)
			return Constants.UNUSED;
		else
			return mean / (double)k;
	}

	
	@Override
	public String parameterToShownText(Object parameter, Object... info) {
		// TODO Auto-generated method stub
		if (parameter == null || !(parameter instanceof ExchangedParameter[]))
			return "";
		
		StringBuffer buffer = new StringBuffer();
		ExchangedParameter[] exParameters = (ExchangedParameter[])parameter;
		for (int i = 0; i < exParameters.length; i++) {
			if (i > 0)
				buffer.append(", ");
			String text = "";
			if (this.rems.get(i) != null)
				text = this.rems.get(i).parameterToShownText(exParameters[i], info);
			buffer.append("{" + text + "}");
		}
		
		return buffer.toString();
	}

	
	@Override
	public synchronized String getDescription() {
		// TODO Auto-generated method stub
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
		
		return buffer.toString();
	}

	
	@Override
	public String getName() {
		// TODO Auto-generated method stub
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "mrem";
	}

	
	@Override
	public Alg newInstance() {
		// TODO Auto-generated method stub
		MutualRegressionEM mrem = new MutualRegressionEM();
		mrem.getConfig().putAll((DataConfig)this.getConfig().clone());
		return mrem;
	}

	
	@Override
	public void setName(String name) {
		// TODO Auto-generated method stub
		getConfig().put(DUPLICATED_ALG_NAME_FIELD, name);
	}

	
	@Override
	public DataConfig createDefaultConfig() {
		// TODO Auto-generated method stub
		DataConfig config = super.createDefaultConfig();
		config.put(R_INDICES_FIELD, R_INDICES_FIELD_DEFAULT);
		config.put(RegressionEMImpl.REM_INVERSE_MODE_FIELD, RegressionEMImpl.REM_INVERSE_MODE_DEFAULT);
		config.put(RegressionEMImpl.REM_BALANCE_MODE_FIELD, RegressionEMImpl.REM_BALANCE_MODE_DEFAULT);
		config.put(MODELS_NUMBER_FIELD, MODELS_NUMBER_DEFAULT);
		
		config.addReadOnly(DUPLICATED_ALG_NAME_FIELD);
		return config;
	}

	
	@Override
	public Object extractResponse(Profile profile) {
		// TODO Auto-generated method stub
		int k = 0;
		double mean = 0;
		for (RegressionEMImpl rem : this.rems) {
			if (rem == null)
				continue;
			
			Object value = rem.extractResponse(profile);
			if (value == null || !(value instanceof Number))
				continue;
			
			double realValue = ((Number)value).doubleValue();
			if (Util.isUsed(realValue)) {
				mean += realValue;
				k++;
			}
		}
		if (k == 0)
			return Constants.UNUSED;
		else
			return mean / (double)k;
	}

	
}
