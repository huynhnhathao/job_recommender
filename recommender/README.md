### How will the demo looks like?

- you login to the main page for the first time, it will ask you to update your cv and your information by a page that look like a gg form.
- the information are: your cv, your age, your relations with other things, all of this is not require in practive but we require it here because we don't want to get into the complicated of creating such a thing
- assume that all the information in your profile is completed at this time, then all the remaining works are recommending jobs to you.
- this is different from the search process where you will enter your requirements and the search tool filter the jobs for you, but rather, here the system recommend jobs to you base on your profile content and behavior.
- there are 4 cases of recommendation, but this project will only considers the first two cases.
	- recommending employers to candidates
	- recommending jobs to candidates
	- candidates to employers
	- candidates to jobs
- the main approach is maybe to precompute every possible thing in advance, and update the parameteres periodically to reduce the inference time.

- the question are, what decide the relation and how do you compare relations?
- similar relations are computed by comparing profile using some nlp algorithm.
- other relation such as like, apply, favorite,.. are assume to be available, provided by the users(this is not the case in practice, but this is a demo!)


- So the first thing must be done is construct the graph, using employers, candidates, jobs and interaction data.
- interaction data are not available, but we can artifially create one.

-after the graph is constructed using all the data, a new candidate join the network by providing her data.
- now the personalized recommendation are extracted from the graph via importance computation approaches.




- Have to decide what are the attributes of a profile for each type of entities. actually, they have already decided that.
- Candidate: age, gender, interest, language, education (diploma, 
major, university), skills, work experience, self-description;
- Employer: country, industry field, scale, description;
- Job: employer, location, requirement, task, opportunity




