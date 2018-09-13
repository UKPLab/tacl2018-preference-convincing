This directory contains data derived from: https://github.com/UKPLab/acl2016-convincing-arguments/ . 
Please see the original repository and article for more information.

# Data description

The `data` directory contains the following sub-folders

* `UKPConvArgCrowdSample-new-CSV`
    * This is a subsample of the full corpus containing the crowdsourced labels in 
    32 tab-delimited csv files, each file corresponding to one debate/side.
* `UKPConvArg1-crowdsample-ranking-CSV`
    * Contains the rank scores for the subsample in UKPConvArgCrowdSample-new-CSV. Contains 
    tab-delimited files with 1,052 arguments with their ID, rank score, and text.
* `UKPConvArgStrict-new-CSV`
    * Cleaned version used for experiments in the article, exported into tab-delimited CSV with ID, more
    convincing argument label (a1 or a2) and both arguments (a1, tab, a2)
* `UKPConvArgAll-new-CSV`
    * This is the full corpus containing the crowdsourced labels in 
    32 tab-delimited csv files, each file corresponding to one debate/side. This contains
    all the preference labels without the cleaning steps used to produce UKPConvArgStrict.
* `UKPConvArg1-Ranking-CSV`
    * Contains the rank scores for all arguments in UKPConvArgAll-new-CSV. Contains 
    tab-delimited files with 1,052 arguments with their ID, rank score, and text.

The data are licensed under CC-BY (Creative Commons Attribution 4.0 International License).

The source arguments originate from
* http://www.createdebate.com licensed under CC-BY (http://creativecommons.org/licenses/by/3.0/)
* http://convinceme.net licensed under Creative Commons Public Domain License
  (https://creativecommons.org/licenses/publicdomain/)

#### CSV

An example from `evolution-vs-creation_evolution.csv`:

```
#id	label	a1	a2
778_80854	a2	Life has been around for ...	Science can prove that ... <br/> The big ...
...
```

* The first line is a comment
* Each line is then a single argument pair, tab-separated
    * Pair ID (first argument ID, second argument ID)
    * Gold label (which one is more convincing)
    * Text of argument 1
    * Text of argument 2
* Line breaks are encoded as `<br/>`
