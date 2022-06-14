(nap1)=

# NAP-1 — Institutional and Funding Partners

```{eval-rst}
:Author: "Juan Nunez-Iglesias <mailto:jni@fastmail.com>"
:Author: "Nicholas Sofroniew <mailto:sofroniewn@gmail.com>"
:Created: '2022-04-21'
:Status: Accepted
:Type: Process
```

## Abstract

Napari has rapidly grown from a small open source library to a massive project
involving over a hundred contributors, and users around the world in academia,
government, and industry. The napari governance, which was modeled after
similar open source projects, does not explicitly deal with funding and
institutional support. This NAP updates the governance document to recognize
institutional and funding support through Institutional and Funding Partners.
The main goal of this NAP is to make clear the benefits of supporting this open
source project where it aligns with an institution's mission.

## Motivation and Scope

Although this NAP predates a formal announcement, napari is in the process of
becoming a NumFOCUS Fiscally Sponsored Project[^NumFOCUS-Fiscal-Sponsorship].
This will allow the project to fundraise directly, receive funding and pay for
developer time. This will assist maintenance and new feature development, among
many other things.

It is therefore important to set out expectations for institutions about the
ways that they can support napari, and the privileges that such support can and
*cannot* provide. For example, it must be clear that napari will remain a
community-run project and that funding alone cannot be used to take over the
vision, mission, or overall direction and decision-making of the project. At
the same time, supporting institutions *may* want a mechanism to ensure that
their priorities are heard and considered by the project.

This NAP specifies the ways funding and in-kind support for the project (for
example, through engineering effort) can benefit institutions.

## Detailed Description

Based on similar documents in other scientific open source projects (see
Related Work, below), this NAP defines Institutional and Funding Partners
(IFPs) and proposes the following changes to our governance:

- restrict the napari Steering Council (SC) size to a minimum of 3 members and
  maximum of 5 members, with a preferred size of 5. (Note: this was in fact the
  intention of the napari founders but it was never explicitly written down.)
- impose the condition that no institution or employer may hold a majority of
  seats at the SC.
- create an Institutional and Funding Partner Advisory Council (IFPAC), made
  up of representatives from institutions providing either funding or in-kind
  support (defined in more detail in the Implementation section).
- allow the IFPAC to elect one member of the napari SC, subject to the
  restrictions above.
- note additional ways in which institutional and funding partners may benefit
  from contributing to the project.

Institutions become IFPs by employing individuals who actively contribute to
napari as part of their official duties, or by committing significant funding
to napari, as determined by the SC.

This NAP proposes the following benefits for IFPs:
- acknowledgement on the napari website, and in talks about the napari project.
- ability to promote their contribution on their own sites and communications.
- ability to improve the project for their specific use cases via their
  engineering, design, and other efforts.
- ability to provide input to the project via the IFPAC and the IFPAC-elected
  SC member.


## Related Work

The model of an Institutional and Funding Partner Advisory Council is inspired
by the [Open Force Field
Consortium](https://openforcefield.org/about/organization/), which supports
open methodologies, open source software, and open data to advance small
molecule simulation for computational drug discovery. The consortium has
attracted significant and diverse collaborators from industry partners. The
Consortium has a seven member Governing Board (similar to our Steering
Council), made up "two elected representatives from supporting Partners and
five Principal Investigators," and an Advisory Board, which includes members
from all Partners that contribute above a threshold level of support. The
Advisory Board provides input on scientific directions and project plans, and
elects industry representatives to the Governing Board.

Another source of inspiration was the Institutional Partner and Funding model
described in the [NumPy governance
document](https://numpy.org/devdocs/dev/governance/governance.html#institutional-partners-and-funding),
which recognizes institutions that significantly contribute to the project, and
encourages institutions explicitly to provide direction to the project via
direct engineering effort by institution employees.

## Implementation

An initial proposal for the additions to the governance is included below:

> The napari steering council (SC) will be no less than three members and no
> more than five members, with a strong preference for an odd number to ensure
> a simple majority vote outcome is always possible, and a preference for five
> members to ensure a diversity of voices. A majority of the SC will not be
> employed by the same entity. One seat on the steering council is reserved for
> a member elected by the Institutional and Funding Partner Advisory Council,
> as detailed below. This member need not be an existing core developer.


>### Institutional and Funding Partners
>
> The SC is the primary leadership body for napari. No outside institution,
> individual or legal entity has the ability to own or control the project
> other than by participating in napari as contributors, core developers, and
> SC members. However, because institutions can be an important source of
> funding and contributions for the project, it is important to formally
> acknowledge institutional participation in the project. We call institutions
> recognized in this way Institutional and Funding Partners.
>
> Institutions become eligible to become an Institutional Partner by employing
> individuals who actively contribute to napari as part of their official
> duties, or by committing significant funding to napari, as determined by the
> SC. Once an institution becomes eligible to become an Institutional Partner,
> the SC must nominate and approve the institution as an Institutional and
> Funding Partner. At that time one individual from the Institutional and
> Funding Partner is expected to become the Institutional and Funding Partner
> Representative and serve on the Institutional and Funding Partner Advisory
> Council. The role of the Institutional and Funding Partner Advisory Council
> is to provide input on project directions and plans, and to elect one
> Institutional and Funding Partner Representative to hold the Institutional
> and Funding Partner seat on the SC.
>
> The Institutional and Funding Partners Advisory Council is expected to
> self-organize according to rules agreed to by the existing Institutional
> Funding Partners. This document does not prescribe how Institutional Funding
> Partners should elect their representative on the napari Steering Council.
> Institutional Funding Partners are expected to work together and with the
> napari community in good faith towards a common goal of improving napari for
> the broader scientific computing community.
>
> If at some point an existing Institutional and Funding Partner is no longer
> contributing any employees or funding, then a one-year grace period
> commences. If during this one-year period they do not contribute any
> employees or funding, then at the end of the period their status as an
> Institutional and Funding Partner will lapse, and resuming it will require
> going through the normal process for new Institutional and Funding Partners.
> If the Institutional and Funding Partner Representative on the SC is from an
> organization that loses its status as an Institutional Partner, that person
> will cease being a member of the SC and the remaining Institutional and
> Funding Partner Advisory Council members may choose a new Representative.
>
> Institutional and Funding Partner benefits are:
>
> - Acknowledgement on the napari website, including homepage, and in talks.
> - Ability to acknowledge their contribution to napari on their own websites
>   and in talks.
> - Ability to provide input to the project through their Institutional Partner
>   Representative.
> - Ability to influence the project through the election of the Institutional
>   and Funding Partner seat on the SC.

The proposed implementation can be found in [PR 4458](https://github.com/napari/napari/pull/4458)

## Backward Compatibility

The steering council will not be immediately changed by this NAP. Once the NAP
is accepted, we aim to formally recognize CZI as the first Institutional and
Funding Partner of the project, and Nicholas Sofroniew's position in the
steering council will become the Institutional and Funding Partner seat.

As noted in the governance changes, should additional institutions come forward
wishing to support napari and become IFPs, they will join CZI in the IFPAC and
may negotiate to elect a different representative.

## Future Work

Future work will add Institutional and Funding Partners according to our
governance.

## Alternatives

Without these changes, institutional support needs to be negotiated ad-hoc with
each partner, which adds a lot of overhead and may result in unequal treatment,
which we want to avoid. It also makes it more difficult a priori for potential
partners to understand the direct benefits of supporting the project, and
therefore make it more difficult to grow the project's support base.

## Discussion

> Does this mean that a non-core developer who joins the SC will also automatically become a core dev?

No, they are merely the IFP SC member.

> Should this document include instructions on how to deal with (or identify) situations where
> IFPs are not working with napari in good faith? How can the community raise concerns about
> and IFP interaction.

Community members can raise their concerns to the SC. Since IFP status is decided by the SC,
it can also be revoked by the SC.

> I think five is a great number for the SC and makes sense, but what about seven?
> Is there a particular reason for the limit of the SC at five?

It was thought to be a good limit with one IFP seat. The open forcefield project has seven members with
two IFP like seats.

> A potential side benefit of IFPs: this provides a reason for employers of existing napari contributors/core-devs to officially recognise work on napari as part of someones 'duties', potentially legitimising work on OSS more generally as 'useful work'

Agreed, the SC can help promote this rationale.

> Should we require/expect organizing rules of the IFP Advisory Council to happen somewhere in the open?

NAP-1 does not prescribe the rules for the IFP Advisory Council, but does ask them to describe their
processes openly.


## References and Footnotes

- [Open Force Field Consortium
  governance](https://openforcefield.org/about/organization/)
- [NumPy project governance and decision-making: Institutional Partners and
  Funding](https://numpy.org/devdocs/dev/governance/governance.html#institutional-partners-and-funding)

## Copyright

This document is dedicated to the public domain with the Creative Commons CC0
license[^id3]. Attribution to this source is encouraged where appropriate, as
per CC0+BY[^id4].

[^id3]: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication,
<https://creativecommons.org/publicdomain/zero/1.0/>

[^id4]: <https://dancohen.org/2013/11/26/cc0-by/>

[^NumFOCUS-Fiscal-Sponsorship]: https://numfocus.org/projects-overview

