#ifndef RF_BOOTSTRAP_VP_H
#define RF_BOOTSTRAP_VP_H
void stackPreDefinedVP(uint    ntree,
                       uint  **bootstrapSizeCase,
                       char ***bootMembershipFlagCase,
                       char ***oobMembershipFlagCase,
                       uint  **oobSizeCase,
                       uint  **ibgSizeCase,
                       uint ***oobMembershipIndexCase,
                       uint ***ibgMembershipIndexCase);
void unstackPreDefinedVP(uint   ntree,
                         uint  *bootstrapSizeCase,
                         char **bootMembershipFlagCase,
                         char **oobMembershipFlagCase,
                         uint  *oobSizeCase,
                         uint  *ibgSizeCase,
                         uint **oobMembershipIndexCase,
                         uint **ibgMembershipIndexCase);
#endif
