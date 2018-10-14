package net.hudup.temp;

import net.hudup.core.Constants;
import net.hudup.core.alg.Alg;
import net.hudup.core.data.DataConfig;
import net.hudup.regression.em.DefaultMixtureREM;

public class DefaultMixtureREM2 extends DefaultMixtureREM {

	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	
	@Override
	public synchronized double executeByXStatistic(double[] xStatistic) {
		if (this.rems == null || this.rems.size() == 0 || xStatistic == null)
			return Constants.UNUSED;
		
		if (xStatistic[1] < 0.5)
			return this.rems.get(0).executeByXStatistic(xStatistic);
		else
			return this.rems.get(1).executeByXStatistic(xStatistic);
//		double result = 0;
//		for (REMImpl rem : this.rems) {
//			ExchangedParameter parameter = rem.getExchangedParameter();
//			
//			double value = rem.executeByXStatistic(xStatistic);
//			if (Util.isUsed(value))
//				result += parameter.getCoeff() * value;
//			else
//				return Constants.UNUSED;
//		}
//		return result;
	}


	@Override
	public synchronized Object execute(Object input) {
		// TODO Auto-generated method stub
		double x = extractRegressorValue(input, 1);
		if (x < 0.5)
			return this.rems.get(0).execute(input);
		else
			return this.rems.get(1).execute(input);
	}


	@Override
	public String getName() {
		// TODO Auto-generated method stub
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "default_mixrem2";
	}

	
	@Override
	public Alg newInstance() {
		// TODO Auto-generated method stub
		DefaultMixtureREM2 mixREM = new DefaultMixtureREM2();
		mixREM.getConfig().putAll((DataConfig)this.getConfig().clone());
		return mixREM;
	}

	
	@Override
	public void setName(String name) {
		// TODO Auto-generated method stub
		getConfig().put(DUPLICATED_ALG_NAME_FIELD, name);
	}


}
