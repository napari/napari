# Abstract

The purpose of this document is to formalize the governance process used by the
`napari` project, to clarify how decisions are made and how the various
elements of our community interact.

This is a consensus-based community project. Anyone with an interest in the
project can join the community, contribute to the project design, and
participate in the decision making process. This document describes how that
participation takes place, how to find consensus, and how deadlocks are
resolved.

# Roles And Responsibilities

## The Community
The napari community consists of anyone using or working with the project
in any way.

## Contributors

A community member can become a contributor by interacting directly with the
project in concrete ways, such as:

- proposing a change to the code via a GitHub pull request;
- reporting issues on our
  [GitHub issues page](https://github.com/napari/napari/issues);
- proposing a change to the documentation, or
  [tutorials](https://github.com/napari/napari-tutorials) via a
  GitHub pull request;
- discussing the design of the napari or its tutorials on the
  [image.sc forum](https://forum.image.sc/tags/napari), or in existing issues and pull
  requests; or
- reviewing
  [open pull requests](https://github.com/napari/napari/pulls),

among other possibilities. Any community member can become a contributor, and
all are encouraged to do so. By contributing to the project, community members
can directly help to shape its future.

Contributors are encouraged to read the [contributing guide](CONTRIBUTING.md).

## Core developers

Core developers are community members that have demonstrated continued
commitment to the project through ongoing contributions. They
have shown they can be trusted to maintain napari with care. Becoming a
core developer allows contributors to merge approved pull requests, cast votes
for and against merging a pull-request, and be involved in deciding major
changes to the API, and thereby more easily carry on with their project related
activities. Core developers appear as organization members on the napari
[GitHub organization](https://github.com/orgs/napari/people). Core
developers are expected to review code contributions while adhering to the
[core developer guide](CORE_DEV_GUIDE.md).

New core developers can be nominated by any existing core developer.
Discussion about new core developer nominations is one of the few activities
that takes place on the project's private management list. The decision to
invite a new core developer must be made by “lazy consensus”, meaning unanimous
agreement by all responding existing core developers. Invitation must take
place at least one week after initial nomination, to allow existing members
time to voice any objections.

## Steering Council

The Steering Council (SC) members are core developers who have additional
responsibilities to ensure the smooth running of the project. SC members are
expected to participate in strategic planning, approve changes to the
governance model, and make decisions about funding granted to the project
itself. (Funding to community members is theirs to pursue and manage.) The
purpose of the SC is to ensure smooth progress from the big-picture
perspective. Changes that impact the full project require analysis informed by
long experience with both the project and the larger ecosystem. When the core
developer community (including the SC members) fails to reach such a consensus
in a reasonable timeframe, the SC is the entity that resolves the issue.

The steering council is fixed in size to XXXXX members. This may be expanded in
the future. The initial steering council of napari consists of XXXXXXXXX.
The SC membership is revisited every January. SC members who do
not actively engage with the SC duties are expected to resign. New members are
added by nomination by a core developer. Nominees should have demonstrated
long-term, continued commitment to the project and its [values](VALUES.md). A
nomination will result in discussion that cannot take more than a month and
then admission to the SC by consensus.

The napari steering council may be contacted at XXXXXXXX.

# Decision Making Process

Decisions about the future of the project are made through discussion with all
members of the community. All non-sensitive project management discussion takes
place on the project XXXXXX and the [issue tracker](https://github.com/napari/napari/issues)
Occasionally, sensitive discussion may occur on a private XXXXXXX.

Decisions should be made in accordance with the [mission, vision and values](VALUES.md)
of the napari project.

napari uses a “consensus seeking” process for making decisions. The group
tries to find a resolution that has no open objections among core developers.
Core developers are expected to distinguish between fundamental objections to a
proposal and minor perceived flaws that they can live with, and not hold up the
decision-making process for the latter.  If no option can be found without
objections, the decision is escalated to the SC, which will itself use
consensus seeking to come to a resolution. In the unlikely event that there is
still a deadlock, the proposal will move forward if it has the support of a
simple majority of the SC.

Decisions (in addition to adding core developers and SC membership as above)
are made according to the following rules:

- **Minor documentation changes**, such as typo fixes, or addition / correction of a
  sentence, require approval by a core developer *and* no disagreement or requested
  changes by a core developer on the issue or pull request page (lazy
  consensus). Core developers are expected to give “reasonable time” to others
  to give their opinion on the pull request if they’re not confident others
  would agree.

- **Code changes and major documentation changes** require agreement by *two*
  core developers *and* no disagreement or requested changes by a core developer
  on the issue or pull-request page (lazy consensus).

- **Changes to the API principles** require a [XXXXXX] and follow the
  decision-making process outlined above.

- **Changes to this governance model or our mission, vision, and values**
  require a [XXXXXXX] and follow the decision-making process outlined above,
  *unless* there is unanimous agreement from core developers on the change.

If an objection is raised on a lazy consensus, the proposer can appeal to the
community and core developers and the change can be approved or rejected by
escalating to the SC.
