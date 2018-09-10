package net.hudup.regression.em;

import static net.hudup.regression.AbstractRegression.extractNumber;
import static net.hudup.regression.em.RegressionEMImpl.R_CALC_VARIANCE_FIELD;

import java.util.List;

import net.hudup.core.Util;
import net.hudup.core.alg.Alg;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.data.DataConfig;
import net.hudup.core.logistic.NextUpdate;

/**
 * This class implements the logistic mixture regression model.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@NextUpdate
public class LogisticMixtureRegressionEM extends AbstractMixtureRegressionEM implements DuplicatableAlg {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	@Override
	protected RegressionEMImpl createRegressionEM() {
		// TODO Auto-generated method stub
		RegressionEMImpl rem = super.createRegressionEM();
		rem.getConfig().put(R_CALC_VARIANCE_FIELD, true);
		return rem;
	}


	@Override
	public Object execute(Object input) {
		// TODO Auto-generated method stub
		if (this.rems == null || this.rems.size() == 0)
			return null;
		
		List<Double> zValues = Util.newList(this.rems.size());
		List<Double> expProbs = Util.newList(this.rems.size());
		double expProbsSum = 0;
		for (int k = 0; k < this.rems.size(); k++) {
			RegressionEMImpl rem = this.rems.get(k);
			double zValue = extractNumber(rem.execute(input));
			if (!Util.isUsed(zValue))
				return null;
			
			zValues.add(zValue);
			
			ExchangedParameter parameter = rem.getExchangedParameter();
			double prob = ExchangedParameter.normalPDF(zValue, 
					parameter.mean(rem.extractRegressors(input)),
					parameter.getZVariance());
			double weight = Math.exp(prob);
			expProbs.add(weight);
			expProbsSum += weight;
		}

		double result = 0;
		for (int k = 0; k < this.rems.size(); k++) {
			result += (expProbs.get(k) / expProbsSum) * zValues.get(k); 
		}
		
		return result;
	}

	
	@Override
	public String getName() {
		// TODO Auto-generated method stub
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "logistic_mixrem";
	}

	
	@Override
	public Alg newInstance() {
		// TODO Auto-generated method stub
		LogisticMixtureRegressionEM logisticMixREM = new LogisticMixtureRegressionEM();
		logisticMixREM.getConfig().putAll((DataConfig)this.getConfig().clone());
		return logisticMixREM;
	}

	
	@Override
	public void setName(String name) {
		// TODO Auto-generated method stub
		getConfig().put(DUPLICATED_ALG_NAME_FIELD, name);
	}

	
}
