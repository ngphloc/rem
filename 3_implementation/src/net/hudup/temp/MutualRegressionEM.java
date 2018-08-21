package net.hudup.temp;

import static net.hudup.regression.AbstractRegression.defaultAttributeList;
import static net.hudup.regression.AbstractRegression.extractNumber;
import static net.hudup.regression.AbstractRegression.splitIndices;
import static net.hudup.temp.RegressionEMImpl.REM_LOOP_BALANCE_MODE_DEFAULT;
import static net.hudup.temp.RegressionEMImpl.REM_LOOP_BALANCE_MODE_FIELD;

import java.util.Arrays;
import java.util.List;

import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.Alg;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Fetcher;
import net.hudup.core.data.MemFetcher;
import net.hudup.core.data.Profile;
import net.hudup.core.parser.TextParserUtil;
import net.hudup.em.ExponentialEM;
import net.hudup.regression.Regression;

/**
 * This class implements expectation maximization (EM) algorithm for many partial regression models.
 * If there are 2 partial models (by default), the mutual regression model becomes dual regression model.
 *  
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@Deprecated
public class MutualRegressionEM extends ExponentialEM implements Regression, DuplicatableAlg {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * List of internal regression model as parameter.
	 */
	protected List<RegressionEMImpl> rems = Util.newList();

	
	/**
	 * List of weights.
	 */
	protected List<Double> weights = Util.newList();
	
	
	@Override
	public Object learn(Object...info) throws Exception {
		// TODO Auto-generated method stub
		if (!prepareInternalData(this.sample)) {
			clearInternalData();
			return null;
		}
		
		ExchangedParameter[] parameters = (ExchangedParameter[])super.learn();
		if (parameters == null) {
			clearInternalData();
			return null;
		}
		
		List<ExchangedParameter> newParameters = Util.newList();
		List<RegressionEMImpl> newRems = Util.newList();
		for (int i = 0; i < parameters.length; i++) {
			ExchangedParameter parameter = parameters[i];
			RegressionEMImpl rem = this.rems.get(i);
			if (parameter != null && rem != null) {
				rem.setParameter(parameter, this.getCurrentIteration());
				newParameters.add(parameter);
				newRems.add(rem);
			}
		}
		if (newParameters.size() == 0) {
			clearInternalData();
			return null;
		}
		
		//Only returning non-null REMs
		parameters = newParameters.toArray(new ExchangedParameter[] {});
		newParameters.clear();
		this.rems.clear();
		this.rems = newRems;
		this.weights = calcResidualWeights(this.rems, this.sample);
		//this.weights = calcUniformWeights(this.rems.size());
		//this.weights = calcRegressWeights(this.rems, this.sample);
		
		return parameters;
	}

	
	/**
	 * Calculating weights based on residuals.
	 * @param rems specified list of regression models.
	 * @param inputSample specified sample.
	 * @return weights.
	 * @throws Exception if any error raises.
	 */
	protected static List<Double> calcResidualWeights(List<RegressionEMImpl> rems, Fetcher<Profile> inputSample) {
		double[] MS = new double[rems.size()];
		Arrays.fill(MS, 0.0);
		double MSTotal = 0;
		List<Double> weights = Util.newList();
		for (int k = 0; k < rems.size(); k++) {
			RegressionEMImpl rem = rems.get(k);
			Fetcher<Profile> estimatedSample = rem.estimate(inputSample, null);
			if (estimatedSample == null)
				return calcUniformWeights(rems.size());
			
			MS[k] = rem.residualMean(estimatedSample);
			if (!Util.isUsed(MS[k]))
				return calcUniformWeights(rems.size()); 
			else if (MS[k] == 0) {
				weights.clear();
				weights.add(0.0);
				for (int j = 0; j < rems.size(); j++) {
					if (j == k)
						weights.add(1.0);
					else
						weights.add(0.0);
				}
				return weights;
			}
			
			MSTotal += MS[k]; 
		}
		
		double[] W = new double[rems.size()];
		Arrays.fill(W, 0.0);
		double WTotal = 0;
		for (int k = 0; k < rems.size(); k++) {
			W[k] = MSTotal / MS[k];
			WTotal += W[k]; 
		}
		
		weights.clear();
		weights.add(0.0);
		for (int k = 0; k < rems.size(); k++) {
			double weight = W[k] / WTotal;
			weights.add(weight);
		}
		
		return weights;
	}
	
	
	/**
	 * Calculating uniform weights.
	 * @param remNumber the number of regression models.
	 * @return uniform weights.
	 */
	protected static List<Double> calcUniformWeights(int remNumber) {
		List<Double> weights0 = Util.newList();
		weights0.add(0.0);
		for (int i = 0; i < remNumber; i++)
			weights0.add(1.0 / (double)remNumber);
		
		return weights0;
	}
	
	
	/**
	 * Calculating weights based on regression model.
	 * @param rems specified list of regression models.
	 * @param inputSample specified sample.
	 * @return weights.
	 * @throws Exception if any error raises.
	 */
	@Deprecated
	protected List<Double> calcRegressWeights(List<RegressionEMImpl> rems, Fetcher<Profile> inputSample) throws Exception {
		List<Double> weights0 = Util.newList();
		weights0.add(0.0);
		for (int i = 0; i < rems.size(); i++)
			weights0.add(1.0 / (double)rems.size());

		//Learning weights by regression model.
		AttributeList attRef = defaultAttributeList(rems.size() + 1);
		List<Profile> profiles = Util.newList();
		while (inputSample.next()) {
			Profile profile = inputSample.pick();
			if (profile == null)
				continue;
			
			Profile newProfile = new Profile(attRef);
			double lastValue = extractNumber(this.extractResponse(profile));
			if (Util.isUsed(lastValue))
				newProfile.setValue(rems.size(), lastValue);
			
			for (int j = 0; j < rems.size(); j++) {
				RegressionEMImpl em = rems.get(j);
				double value = extractNumber(em.execute(profile));
				if (Util.isUsed(value))
					newProfile.setValue(j, value);
			}
			
			boolean missing = false;
			for (int j = 0; j < attRef.size(); j++) {
				if (newProfile.isMissing(j)) {
					missing = true;
					break;
				}
			}
			if (!missing)
				profiles.add(newProfile);
		}
		inputSample.reset();
		
		if (profiles.size() == 0)
			return weights0;
		
		StringBuffer indices = new StringBuffer();
		for (int i = 0; i < attRef.size(); i++) {
			if (i > 0)
				indices.append(", ");
			indices.append(i);
		}
		MemFetcher<Profile> weightsSample =  new MemFetcher<>(profiles);
		RegressionEMImpl weightsEM = new RegressionEMImpl();
		DataConfig config = weightsEM.getConfig();
		config.put(R_INDICES_FIELD, indices.toString());
		config.put(REM_LOOP_BALANCE_MODE_FIELD, getConfig().get(REM_LOOP_BALANCE_MODE_FIELD));
		weightsEM.setup(weightsSample);
		
		if (weightsEM.getParameter() == null)
			return weights0;

		double[] alpha = ((ExchangedParameter)weightsEM.getParameter()).getVector();
		List<Double> weights = Util.newList();
		for (int j = 0; j < alpha.length; j++)
			weights.add(alpha[j]);
			
		weightsEM.unsetup();
		weightsEM.clearInternalData();
		weightsSample.close();
		
		return weights;
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
			for (int i = 0; i < attList.size(); i++) {
				if (i > 0)
					indices.append(", ");
				indices.append(i);
			}
			indicesList.add(indices.toString());
			
			if (attList.size() > 2) {
				for (int i = 0; i < attList.size() - 1; i++) {
					indicesList.add(i + ", " + (attList.size() - 1));
				}
			}
		}
			
		for (int i = 0; i < indicesList.size(); i++) {
			RegressionEMImpl rem = new RegressionEMImpl() {

				/**
				 * Serial version UID for serializable class.
				 */
				private static final long serialVersionUID = 1L;

				@Override
				public synchronized Object learn(Object...info) throws Exception {
					// TODO Auto-generated method stub
					boolean prepared = prepareInternalData(inputSample);
					if (prepared)
						return prepared;
					else
						return null;
				}

			};
			
			DataConfig config = rem.getConfig();
			config.put(R_INDICES_FIELD, indicesList.get(i));
			config.put(REM_LOOP_BALANCE_MODE_FIELD, thisConfig.get(REM_LOOP_BALANCE_MODE_FIELD));
			rem.setup(inputSample);
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
		this.weights.clear();
	}

	
	@Override
	protected Object expectation(Object currentParameter, Object...info) throws Exception {
		// TODO Auto-generated method stub
		ExchangedParameter[] exParameters = (ExchangedParameter[])currentParameter;
		
		ExchangedParameter[] stats = new ExchangedParameter[exParameters.length];
		Arrays.fill(stats, null);
		List<ExchangedParameter> successStats = Util.newList();
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
		List<ExchangedParameter> successStats2 = Util.newList();
		for (ExchangedParameter stat : successStats) {
			if (stat.vector.length == N)
				successStats2.add(stat);
		}
		if (successStats2.size() <= 1)
			return stats;
		
		//Estimating Z based on same-length Z statistics
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
	protected Object maximization(Object currentStatistic, Object...info) throws Exception {
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
	protected boolean terminatedCondition(Object estimatedParameter, Object currentParameter, Object previousParameter, Object... info) {
		// TODO Auto-generated method stub
		ExchangedParameter[] exEstimatedParameters = (ExchangedParameter[])estimatedParameter;
		ExchangedParameter[] exCurrentParameters = (ExchangedParameter[])currentParameter;
		ExchangedParameter[] exPreviousParameters = (ExchangedParameter[])previousParameter;

		boolean terminated = true;
		for (int i = 0; i < this.rems.size(); i++) {
			RegressionEMImpl rem = this.rems.get(i);
			ExchangedParameter exEstimatedParameter = exEstimatedParameters[i];
			ExchangedParameter exCurrentParameter = exCurrentParameters[i];
			ExchangedParameter exPreviousParameter = exPreviousParameters[i];
			
			if (rem == null || exCurrentParameter == null || exEstimatedParameter == null)
				continue;
			
			terminated = terminated && rem.terminatedCondition(exEstimatedParameter, exCurrentParameter, exPreviousParameter, info);
			if (!terminated)
				return false;
		}
		
		return terminated;
	}

	
	@Override
	public synchronized Object execute(Object input) {
		// TODO Auto-generated method stub
		double result = this.weights.get(0);
		for (int i = 0; i < this.rems.size(); i++) {
			RegressionEMImpl rem = this.rems.get(i);
			
			double value = extractNumber(rem.execute(input));
			if (Util.isUsed(value))
				result += this.weights.get(i + 1) * value;
			else
				return null;
		}
		return result;
	}

	
	@Override
	public synchronized String parameterToShownText(Object parameter, Object... info) {
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
		
		String weightsText = TextParserUtil.toText(this.weights, ",");
		buffer.append(" : [" + weightsText + "]");
		
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
		
		String weightsText = TextParserUtil.toText(this.weights, ",");
		buffer.append(" : [" + weightsText + "]");

		return buffer.toString();
	}

	
	@Override
	public String getName() {
		// TODO Auto-generated method stub
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "mutual_rem";
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
		config.put(REM_LOOP_BALANCE_MODE_FIELD, REM_LOOP_BALANCE_MODE_DEFAULT);
		
		config.addReadOnly(DUPLICATED_ALG_NAME_FIELD);
		return config;
	}

	
	@Override
	public synchronized Object extractResponse(Profile profile) {
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
