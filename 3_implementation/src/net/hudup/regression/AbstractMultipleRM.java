package net.hudup.regression;

import java.util.Arrays;
import java.util.List;

import net.hudup.core.Util;
import net.hudup.core.alg.AbstractTestingAlg;
import net.hudup.core.alg.SetupAlgEvent;
import net.hudup.core.alg.SetupAlgEvent.Type;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Dataset;
import net.hudup.core.data.Fetcher;
import net.hudup.core.data.Profile;


/**
 * This is the most abstract class for multiple regression model. It implements partially the interface {@link RM}.
 * 
 * @author Loc Nguyen
 * @version 1.0*
 */
public abstract class AbstractMultipleRM extends AbstractTestingAlg implements MultipleRM {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * List of internal regression model as parameter.
	 */
	protected List<RM> regressions = Util.newList(); 
	
	
	@Override
	public void setup(Dataset dataset, Object... info) throws Exception {
		// TODO Auto-generated method stub
		List<Object> additionalInfo = Util.newList();
		List<RM> regressions = Util.newList();
		for (Object el : info) {
			if (el instanceof RM)
				regressions.add((RM)el);
			else
				additionalInfo.add(el);
		}
		this.setup(dataset, additionalInfo.toArray(), regressions.toArray(new RM[] {}));
	}


	@Override
	public void setup(Fetcher<Profile> sample, Object... info) throws Exception {
		// TODO Auto-generated method stub
		List<Object> additionalInfo = Util.newList();
		List<RM> regressions = Util.newList();
		for (Object el : info) {
			if (el instanceof RM)
				regressions.add((RM)el);
			else
				additionalInfo.add(el);
		}

		this.setup(sample, additionalInfo.toArray(), regressions.toArray(new RM[] {}));
	}


	/**
	 * Setting up this multiple regression algorithm based on specified dataset and many partial regressions.
	 * @param dataset specified dataset.
	 * @param info additional parameters to set up this algorithm.
	 * @param regressions many specified partial regressions. 
	 * @throws Exception if any error raises.
	 */
	@SuppressWarnings("unchecked")
	public synchronized void setup(Dataset dataset, Object[] info, RM...regressions) throws Exception {
		// TODO Auto-generated method stub
		unsetup();
		for (RM regression : regressions) {
			this.regressions.add(regression);
		}
		if (this.regressions.size() == 0)
			return;

		this.dataset = dataset;
		if (info != null && info.length > 0 && (info[0] instanceof Fetcher<?>))
			this.sample = (Fetcher<Profile>)info[0];
		else
			this.sample = dataset.fetchSample();
		
		List<String> cfgIndicesList = AbstractRM.splitIndices(
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
	public void setup(Fetcher<Profile> sample, Object[] info, RM...regressions) throws Exception {
		// TODO Auto-generated method stub
		List<Object> additionalInfo = Util.newList();
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
	public Object learn(Object...info) throws Exception {
		// TODO Auto-generated method stub
		List<Object> parameterList = Util.newList();
		boolean success = false;
		for (RM regression : this.regressions) {
			regression.setup(this.sample);
			Object parameter = regression.getParameter();
			parameterList.add(parameter);
			if (parameter != null)
				success = true;
		}
		if (!success)
			return null;
		else
			return parameterList; 
	}

	
	@Override
	public synchronized Object execute(Object input) {
		// TODO Auto-generated method stub
		List<Object> resultList = Util.newList();
		boolean success = false;
		for (RM regression : this.regressions) {
			Object result = regression.execute(input);
			resultList.add(result);
			if (result != null)
				success = true;
		}
		
		if (!success)
			return null;
		else
			return resultList;
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
		List<Object> parameterList = Util.newList();
		boolean success = false;
		for (RM regression : this.regressions) {
			Object parameter = regression.getParameter();
			parameterList.add(parameter);
			if (parameter != null)
				success = true;
		}
		
		if (!success)
			return null;
		else
			return parameterList;
	}

	
	@Override
	public DataConfig createDefaultConfig() {
		// TODO Auto-generated method stub
		DataConfig config = super.createDefaultConfig();
		config.put(R_INDICES_FIELD, R_INDICES_DEFAULT);
		return config;
	}

	
	@Override
	public synchronized String parameterToShownText(Object parameter, Object... info) {
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
			buffer.append("{" + text + "}");
		}
		
		return buffer.toString();
	}

	
	@Override
	public synchronized String getDescription() {
		// TODO Auto-generated method stub
		StringBuffer buffer = new StringBuffer();
		
		for (int i = 0; i < this.regressions.size(); i++) {
			RM regression = this.regressions.get(i);
			if (i > 0)
				buffer.append(", ");
			String text = regression.getDescription();
			buffer.append("{" + text + "}");
		}
		
		return buffer.toString();
	}


	@Override
	public synchronized Object extractResponse(Object input) {
		// TODO Auto-generated method stub
		List<Object> valueList = Util.newList();
		boolean success = false;
		for (RM regression : this.regressions) {
			Object value = regression.extractResponse(input);
			valueList.add(value);
			if (value != null)
				success = true;
		}
		
		if (!success)
			return null;
		else
			return valueList;
	}

	
}
