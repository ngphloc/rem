package net.hudup.regression;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import net.hudup.core.alg.AbstractTestingAlg;
import net.hudup.core.alg.SetupAlgEvent;
import net.hudup.core.alg.SetupAlgEvent.Type;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Dataset;
import net.hudup.core.data.Fetcher;
import net.hudup.core.data.Profile;


/**
 * This is the most abstract class for multiple regression model. It implements partially the interface {@link Regression}.
 * 
 * @author Loc Nguyen
 * @version 1.0*
 */
public abstract class AbstractMultipleRegression extends AbstractTestingAlg implements MultipleRegression {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * List of internal regression model as parameter.
	 */
	protected List<Regression> regressions = new ArrayList<>(); 
	
	
	@Override
	public void setup(Dataset dataset, Object... info) throws Exception {
		// TODO Auto-generated method stub
		List<Object> additionalInfo = new ArrayList<>();
		List<Regression> regressions = new ArrayList<>();
		for (Object el : info) {
			if (el instanceof Regression)
				regressions.add((Regression)el);
			else
				additionalInfo.add(el);
		}
		this.setup(dataset, additionalInfo.toArray(), regressions.toArray(new Regression[] {}));
	}


	@Override
	public void setup(Fetcher<Profile> sample, Object... info) throws Exception {
		// TODO Auto-generated method stub
		List<Object> additionalInfo = new ArrayList<>();
		List<Regression> regressions = new ArrayList<>();
		for (Object el : info) {
			if (el instanceof Regression)
				regressions.add((Regression)el);
			else
				additionalInfo.add(el);
		}

		this.setup(sample, additionalInfo.toArray(), regressions.toArray(new Regression[] {}));
	}


	/**
	 * Setting up this multiple regression algorithm based on specified dataset and many partial regressions.
	 * @param dataset specified dataset.
	 * @param info additional parameters to set up this algorithm.
	 * @param regressions many specified partial regressions. 
	 * @throws Exception if any error raises.
	 */
	@SuppressWarnings("unchecked")
	public synchronized void setup(Dataset dataset, Object[] info, Regression...regressions) throws Exception {
		// TODO Auto-generated method stub
		unsetup();
		for (Regression regression : regressions) {
			this.regressions.add(regression);
		}
		if (this.regressions.size() == 0)
			return;

		this.dataset = dataset;
		if (info != null && info.length > 0 && (info[0] instanceof Fetcher<?>))
			this.sample = (Fetcher<Profile>)info[0];
		else
			this.sample = dataset.fetchSample();
		
		List<String> cfgIndicesList = AbstractRegression.splitIndices(
				this.getConfig().getAsString(R_INDICES_FIELD));
		for (int i = 0; i < Math.min(cfgIndicesList.size(), this.regressions.size()); i++) {
			this.regressions.get(i).getConfig().put(R_INDICES_FIELD, cfgIndicesList.get(i));
		}
		
		learn();
		
		SetupAlgEvent evt = new SetupAlgEvent(
				this,
				Type.done,
				this,
				dataset,
				"Learned models: " + this.getDescription());
		fireSetupEvent(evt);
	}

	
	/**
	 * Setting up this multiple regression algorithm based on specified sample and many partial regressions.
	 * @param sample specified sample.
	 * @param info additional parameters to set up this algorithm.
	 * @param regressions many specified partial regressions. 
	 * @throws Exception if any error raises.
	 */
	public void setup(Fetcher<Profile> sample, Object[] info, Regression...regressions) throws Exception {
		// TODO Auto-generated method stub
		List<Object> additionalInfo = new ArrayList<>();
		additionalInfo.add(sample);
		additionalInfo.addAll(Arrays.asList(info));
		
		this.setup((Dataset)null, additionalInfo.toArray(), regressions);
	}

	
	@Override
	public synchronized void unsetup() {
		// TODO Auto-generated method stub
		super.unsetup();
		this.regressions.clear();
	}


	@Override
	public Object learn() throws Exception {
		// TODO Auto-generated method stub
		List<Object> parameterList = new ArrayList<>();
		for (Regression regression : this.regressions) {
			regression.setup(this.sample);
			Object parameter = regression.getParameter();
			if (parameter != null)
				parameterList.add(parameter);
		}
		if (parameterList.size() == 0)
			return null;
		else
			return parameterList.size() == this.regressions.size() ? parameterList : null; // The learn() method is successful if all internal regressions are learned successful. 
	}

	
	@Override
	public Object execute(Object input) {
		// TODO Auto-generated method stub
		List<Object> resultList = new ArrayList<>();
		for (Regression regression : this.regressions) {
			Object result = regression.execute(input);
			resultList.add(result);
		}
		
		if (resultList.size() == 0)
			return null;
		else
			return resultList;
	}

	
	@Override
	public Object getParameter() {
		// TODO Auto-generated method stub
		List<Object> parameterList = new ArrayList<>();
		for (Regression regression : this.regressions) {
			Object parameter = regression.getParameter();
			parameterList.add(parameter);
		}
		
		if (parameterList.size() == 0)
			return null;
		else
			return parameterList;
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
		if (parameter == null || !(parameter instanceof List<?>))
			return "";
		
		StringBuffer buffer = new StringBuffer();
		List<?> parameterList = (List<?>)parameter;
		for (int i = 0; i < parameterList.size(); i++) {
			Object element = parameterList.get(i);
			if (i > 0)
				buffer.append(", ");
			String text = this.regressions.get(i).parameterToShownText(element, info);
			buffer.append("(" + text + ")");
		}
		
		return buffer.toString();
	}

	
	@Override
	public String getDescription() {
		// TODO Auto-generated method stub
		StringBuffer buffer = new StringBuffer();
		
		for (int i = 0; i < this.regressions.size(); i++) {
			Regression regression = this.regressions.get(i);
			if (i > 0)
				buffer.append(", ");
			String text = regression.getDescription();
			buffer.append("{" + text + "}");
		}
		
		return buffer.toString();
	}


	@Override
	public Object extractResponse(Profile profile) {
		// TODO Auto-generated method stub
		List<Object> valueList = new ArrayList<>();
		for (Regression regression : this.regressions) {
			Object value = regression.extractResponse(profile);
			valueList.add(value);
		}
		
		if (valueList.size() == 0)
			return null;
		else
			return valueList;
	}

	
}
