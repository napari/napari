:orphan: 
  
 (nap-8)= 
  
 # NAP-8 — Telemetry
  
 ```{eval-rst} 
 :Author: Grzegorz Bokota 
 :Created: 2023-08-11
 :Resolution: <url> (required for Accepted | Rejected | Withdrawn) 
 :Resolved: <date resolved, in yyyy-mm-dd format> 
 :Status: Draft
 :Type: Standards Track 
 :Version effective: <version-number> (for accepted NAPs) 
 ``` 
  
 ## Abstract 
  
This NAP describes how telemetry would be used by the napari project and the architecture
and solutions proposed to maximize the privacy of our users. 
 
 ## Motivation and Scope

With the growth of napari,
the standard feedback loop through napari community meetings and napari-related events at conferences has reached its capacity.
Also, we collect many feature requests for which we cannot find volunteers for implementation. 

To have the possibility of sustainable development of the project, 
we will either need to rely on paid contractors or on companies donating employee time managed by the core devs.

Both scenarios require us to provide some information about the estimated number of users to prove to potential 
funders that their donation/grant will be used in a valuable way. 

Adding the option for monitoring plugin usage allows us to identify heavily used plugins and try
to establish cooperation with their maintainers
to reduce the probability that the plugin will not be ready for a new napari release.
Such monitoring could contain not only the list of installed plugins
but also which commands and contributions are used most often.

Also collecting information about data types and their size will provide valuable information about the typical use cases of napari.

Still, users need to be able to opt out of such monitoring,
and adjust the level of detail of the information that is sent to the napari server.
Each time when we update the collected data,
we should inform users about the changes and provide them with the possibility to opt out of telemetry.

Users could also provide a temporary agreement for sending telemetry. 
Then after a given period of time, the dialog with question will be shown again.
 
  
 ## Detailed Description

`napari-telemetry` will be a package responsible for collecting and sending telemetry data to the napari server.
It will be installed after user confirmation.
It will contain callbacks for data collection, and utils for storage and sending.
Also, this package will contain utils for validating if the user has agreed to telemetry. 

In the main package, there is a need to add code to ask users if they want to enable telemetry.
This code should be executed only once per environment.

Telemetry should contain the following ways to disable it:

1. Disable in settings
2. Uninstall `napari-telemetry` package
3. Environment variable `NAPARI_TELEMETRY=0`
4. System-wide disablement, e.g., via firewall filtering for hpc or other environments.

The user should be able to adjust the telemetry level of detail. The following levels are proposed:

1. `none` - no telemetry is collected
2. `basic` - information about the napari version, python version, OS, and CPU architecture is collected, 
   and if it is the first report by the user.
   There is also a user identifier created based on computer details
   that will be regenerated each week to prevent tracking the user,
   but allow us to accurately gauge individual user numbers. 
3. `middle` - same as in `basic` plus information about the list of installed plugins and their versions are also collected. 
   We take care to not collect data about plugins that are not intended to be public,
   so we will only collect information about plugins searchable as using plugin dialog or napari hub.
   We also will not collect information about plugins that are installed in a non-stable version.
4. `full` - same as in `middle`
   plus collects information about plugin usage by binding to app-model and logging plugin commands used. 
   Additionally basic information about data like types
   (`np.ndarray`, `dask.array`, `zarr.Array`, etc.) and its size will be collected.

There should be a visible indicator that telemetry is enabled (for example, on the status bar). 

The second part of this work should be to set up the server to collect telemetry data.
After collecting data, 
it should provide a basic public dashboard that will allow the community to see aggregated information.

We propose the following data retention policy:

1) Up to 2 weeks for logs.
2) Up to 2 months of raw data (1 month of collection, then aggregation and time to validate aggregated data).
3) Infinite of aggregated data.
  
## Privacy assessment

During the preparation of this NAP, we assume that none of the collected data will be presented in 
a form that allows to identify a single user or identify a research area of user.
We also select a set of data that will be collected to minimize the possibility of revealing sensitive data.
However, it is impossible to guarantee that it will not be possible to identify a single user
(for example, by checking installed plugin combinations).

Because of this, we propose to not publish raw data and only show aggregated results.
The aggregation will be performed using scripts. 
Napari core devs will access raw data only if there are errors in the aggregation process.

We also will publish a list of endpoints for each level of telemetry, 
so the given level of telemetry could be blocked on the organization level 
(for example, by the rule on the firewall).


If someone found that we are publishing some problematic data, we will remove them 
and update the aggregation process to prevent such a situation in the future.
This NAP will be updated to reflect the current state of telemetry. 

 
## Related Work 

Total systems:
https://plausible.io/
https://sentry.io/
https://opentelemetry.io/

Visualizations:
https://github.com/grafana/grafana
  

  
## Implementation 

The key consideration for implementation should be the low cost of maintenance. 
So the solution should be as simple as possible.
We could either use existing solutions on the server side or implement our own.

The benefit of existing solutions is that most of the work is already done.
The downside is that it may require additional cost of maintenance.
This cost may be caused by many features that are not needed for napari and could increase the risk of leaking data.
Quick checks of their code revealed they are implemented in techniques that are not familiar to napari core devs.
So, if we decide to use them, we should select an SAS solution that will be maintained by the company.


For now, I suggest creating a simple REST API server for collecting the data. 
It could be a simple Python FastAPI server that will store data in the SQLite database.
Connection to server will be encrypted using HTTPS and certificate provided by LetsEncrypt.

Data for aggregation should be extracted from the database using a script running on the same machine.

The output of the aggregation script should be loaded to some existing visualization tool, like grafana.

It may be nice to host raw and aggregate data on separate servers —
then even if the data presented on the dashboard is compromised,
the raw data will be not exposed to the world.

Having both server and aggregation scripts in Python will reduce maintenance costs for napari core devs.

We should register the `telemetry.napari.org` domain and use it for the server. 
The main page will contain this NAP and a link to the summary dashboard.


The main part of the application side should be implemented in `napari-telemetry` package. 
The package should not report in stream mode, but collect data on the disk and send it in batches.
This will reduce the risk of leaking data.
The package should implement a utility to allow users to preview collected data before sending it to the server.

In napari itself, the following changes should be implemented:

1) The indicator that shows the telemetry status
2) The dialog that asks a user if they want to enable telemetry
3) code to check if telemetry is enabled (to not load the `napari-telemetry` package if it is disabled)
4) code required to init `napari-telemetry` package


## GDPR compliance

I'm almost sure that we will not collect data that are covered by GDPR. 
But to get a better atmosphere, 
we need to add instruction how a user could retrieve their unique identifier and set up a process 
for requests to remove data from the server. 
It is not a high probability of usage as the life span of data is short,
but we need to be prepared for such a situation. I suggest to use e-mail for that.


  
## Backward Compatibility 
  
 Not relevant

## Future Work 

A nice extension may be the ability for the steering council to create a certificate of telemetry output that could be
given to plugin maintainers to prove to supervisors that their plugin is used by the community. 

  
## Alternatives 

During the discussion, there is a proposal to use the same approach as used in ImageJ. 
 
This would mean, instead of implementing telemetry on the client side, we could implement it on the update server side.
The advantage and disadvantage of such a solution is that no user could opt out of telemetry.
Also, such a method could potentially provide information about the Python version,
napari version, and list of installed plugins.
All others will require a mechanism from this NAP.

It will also require updates on the Napari side
as currently we only communicate with the update server when a user opens the plugin manager.
Also,
to have proper information about installed plugins, we will need
to send information about the list of installed plugins
instead of just downloading the information about all plugins from the server. 

As this solution provides less information, 
it does not allow for opt-out and could cause ban-listing of the update server IP address,
I do not recommend it.

But based on talks
that happen during the discussion, we may think about more frequent checks for updates
to inform users that they could update their Napari or plugin version.
For such a change, we need to update our update server to provide information per Python version
(as some plugins could drop old Python earlier).

The second alternative is use a third-party solution like [plausible.io](https://plausible.io/). 
But from my perspective, 
it is harder to adjust a set of data that is collected as these services are designed to monitor webpages. 

  
 ## Discussion 
  
 This section may just be a bullet list including links to any discussions 
 regarding the NAP, but could also contain additional comments about that 
 discussion: 
  
 - This includes links to discussion forum threads or relevant GitHub discussions. 
  
 ## References and Footnotes 
  
 All NAPs should be declared as dedicated to the public domain with the CC0 
 license [^id3], as in `Copyright`, below, with attribution encouraged with 
 CC0+BY [^id4]. 
  
 [^id3]: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication, 
     <https://creativecommons.org/publicdomain/zero/1.0/> 
  
 [^id4]: <https://dancohen.org/2013/11/26/cc0-by/> 
  
 ## Copyright 
  
 This document is dedicated to the public domain with the Creative Commons CC0 
 license [^id3]. Attribution to this source is encouraged where appropriate, as per 
 CC0+BY [^id4].