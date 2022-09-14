(napari-governance)=
# Governance model

## Abstract

The purpose of this document is to formalize the governance process used by the
`napari` project, to clarify how decisions are made and how the various
elements of our community interact.

This is a consensus-based community project. Anyone with an interest in the
project can join the community, contribute to the project design, and
participate in the decision making process. This document describes how that
participation takes place, how to find consensus, and how deadlocks are
resolved.

## Roles and responsibilities

### The community

The napari community consists of anyone using or working with the project
in any way.

### Contributors

A community member can become a contributor by interacting directly with the
project in concrete ways, such as:

- proposing a change to the code via a GitHub
  [GitHub pull request](https://github.com/napari/napari/pulls);
- reporting issues on our
  [GitHub issues page](https://github.com/napari/napari/issues);
- proposing a change to the documentation, or
  tutorials via a [GitHub pull request](https://github.com/napari/napari/pulls);
- discussing the design of the napari or its tutorials on in existing
  [issues](https://github.com/napari/napari/issues) and
  [pull requests](https://github.com/napari/napari/pulls);
- discussing examples or use cases on the
  [image.sc forum](https://forum.image.sc/tag/napari) under the #napari tag; or
- reviewing [open pull requests](https://github.com/napari/napari/pulls)

among other possibilities. Any community member can become a contributor, and
all are encouraged to do so. By contributing to the project, community members
can directly help to shape its future.

Contributors are encouraged to read the [contributing guide](napari-contributing).

### Core developers

Core developers are community members that have demonstrated continued
commitment to the project through ongoing contributions. They
have shown they can be trusted to maintain napari with care. Becoming a
core developer allows contributors to merge approved pull requests, cast votes
for and against merging a pull-request, and be involved in deciding major
changes to the API, and thereby more easily carry on with their project related
activities. Core developers appear as organization members on the napari
[GitHub organization](https://github.com/orgs/napari/people) and are on our
[@napari/core-devs](https://github.com/orgs/napari/teams/core-devs) GitHub team. Core
developers are expected to review code contributions while adhering to the
[core developer guide](core-dev-guide). New core developers can be nominated
by any existing core developer, and for details on that process see our core
developer guide. For a full list of core developers see our [About Us](team) page.

### Steering council

The Steering Council (SC) members are primarily core developers who have additional
responsibilities to ensure the smooth running of the project. SC members are
expected to participate in strategic planning, approve changes to the
governance model, and make decisions about funding granted to the project
itself. (Funding to community members is theirs to pursue and manage). The
purpose of the SC is to ensure smooth progress from the big-picture
perspective. Changes that impact the full project require analysis informed by
long experience with both the project and the larger ecosystem. When the core
developer community (including the SC members) fails to reach such a consensus
in a reasonable timeframe, the SC is the entity that resolves the issue.

Members of the SC also have the "owner" role within the [napari GitHub organization](https://github.com/napari)
and are ultimately responsible for managing the napari GitHub account, the [@napari_imaging](https://twitter.com/napari_imaging)
twitter account, the [napari website](https://napari.org), and other similar napari owned resources.

The SC will be no less than three members and no more than five members,
with a strong preference for an odd number to ensure a simple majority vote
outcome is always possible, and a preference for five members to ensure a
diversity of voices. All deadlocked votes of the SC will be postponed until
there is an odd number of members and another vote can be held. A majority of the
SC will not be employed by the same entity. One seat on the SC is reserved
for a member elected by the [Institutional and Funding Partner Advisory Council](#institutional-and-funding-partners),
as detailed below. This member need not be an existing core developer.

The SC membership, including the Institutional and Funding Partner (IFP) seat, is revisited every January.
SC members who do not actively engage with the SC duties are expected to resign. New members for 
vacant spots are added by nomination by a core developer. Nominees should have demonstrated
long-term, continued commitment to the project and its [mission and values](mission-and-values). A
nomination will result in discussion that cannot take more than a month and
then admission to the SC by consensus. During that time deadlocked votes of the SC will
be postponed until the new member has joined and another vote can be held. The IFP seat
is elected by the IFP Advisory Council.

The SC may be contacted at `napari-steering-council@googlegroups.com`.
Or via the [@napari/steering-council](https://github.com/orgs/napari/teams/steering-council) GitHub team. For a list of the current SC see our [About Us](team) page.

### Institutional and funding partners

The SC is the primary leadership body for napari. No outside institution,
individual or legal entity has the ability to own or control the project
other than by participating in napari as contributors, core developers, and
SC members. However, because institutions can be an important source of
funding and contributions for the project, it is important to formally
acknowledge institutional participation in the project. We call institutions
recognized in this way Institutional and Funding Partners (IFPs).

Institutions become eligible to become an IFP by employing individuals
who actively contribute to napari as part of their official
duties, or by committing significant funding to napari, as determined by the
SC. Once an institution becomes eligible to become an IFP, the SC must
nominate and approve the institution as an IFP. At that time one individual
from the IFP is expected to become the IFP Representative and serve on the
IFP Advisory Council. The role of the IFP Advisory Council is to provide
input on project directions and plans, and to elect one IFP Representative
to hold the IFP seat on the SC.

The IFP Advisory Council is expected to self-organize according to rules
agreed to by the existing IFPs. This document does not prescribe how IFPs
should elect their representative on the SC, though the IFP Advisory Council
should describe this process openly. IFPs are expected to work together
and with the napari community in good faith towards a common goal
of improving napari for the broader scientific computing community.

If at some point an existing IFP is no longer contributing any employees
or funding, then a one-year grace period commences. If during this one-year
period they do not contribute any employees or funding, then at the end of
the period their status as an IFP will lapse, and resuming it will require
going through the normal process for new IFPs. If the IFP Representative on
the SC is from an organization that loses its status as an Institutional Partner,
that person will cease being a member of the SC and the remaining IFP Advisory
Council members may choose a new Representative at their earliest convenience.

IFP benefits are:

- Acknowledgement on the napari website, including homepage, and in talks.
- Ability to acknowledge their contribution to napari on their own websites and in talks.
- Ability to provide input to the project through their Institutional Partner
Representative.
- Ability to influence the project through the election of the Institutional
and Funding Partner seat on the SC.

For a full list of current IFPs and their Representatives see our [About Us](team) page.

## Decision making process

Decisions about the future of the project are made through discussion with all
members of the community. All non-sensitive project management discussion takes
place on the [issue tracker](https://github.com/napari/napari/issues) and project
[zulip](https://napari.zulipchat.com/) community chat channel. Occasionally,
sensitive discussion may occur on a private core developer mailing list
`napari-core-devs@googlegroups.com` or private chat channel.

Decisions should be made in accordance with the [mission and values](mission-and-values)
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

- **Code changes and major documentation changes** require agreement by *one*
  core developer *and* no disagreement or requested changes by a core developer
  on the issue or pull-request page (lazy consensus). For all changes of this type,
  core developers are expected to give “reasonable time” after approval and before
  merging for others to weigh in on the pull request in its final state.

- **Changes to the API principles** require a dedicated issue on our
  [issue tracker](https://github.com/napari/napari/issues) and follow the
  decision-making process outlined above.

- **Changes to this governance model or our mission, vision, and values**
  require a  dedicated issue on our [issue tracker](https://github.com/napari/napari/issues)
  and follow the decision-making process outlined above,
  *unless* there is unanimous agreement from core developers on the change in
  which case it can move forward faster.

If an objection is raised on a lazy consensus, the proposer can appeal to the
community and core developers and the change can be approved or rejected by
escalating to the SC.