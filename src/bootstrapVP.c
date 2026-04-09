
// *** THIS HEADER IS AUTO GENERATED. DO NOT EDIT IT ***
#include           "shared/globalCore.h"
#include           "shared/externalCore.h"
#include           "global.h"
#include           "external.h"

// *** THIS HEADER IS AUTO GENERATED. DO NOT EDIT IT ***

      
    

#include "shared/nrutil.h"
void stackPreDefinedVP(uint    ntree,
                       uint  **bootstrapSizeCase,
                       char ***bootMembershipFlagCase,
                       char ***oobMembershipFlagCase,
                       uint  **oobSizeCase,
                       uint  **ibgSizeCase,
                       uint ***oobMembershipIndexCase,
                       uint ***ibgMembershipIndexCase) {
  *bootstrapSizeCase = uivector(1, ntree);
  *bootMembershipFlagCase = (char **) new_vvector(1, ntree, NRUTIL_CPTR);
  *oobMembershipFlagCase = (char **) new_vvector(1, ntree, NRUTIL_CPTR);
  *oobSizeCase = uivector(1, ntree);
  *ibgSizeCase = uivector(1, ntree);
  *oobMembershipIndexCase = (uint **) new_vvector(1, ntree, NRUTIL_UPTR);
  *ibgMembershipIndexCase = (uint **) new_vvector(1, ntree, NRUTIL_UPTR);
}
void unstackPreDefinedVP(uint   ntree,
                         uint  *bootstrapSizeCase,
                         char **bootMembershipFlagCase,
                         char **oobMembershipFlagCase,
                         uint  *oobSizeCase,
                         uint  *ibgSizeCase,
                         uint **oobMembershipIndexCase,
                         uint **ibgMembershipIndexCase) {
  free_uivector(bootstrapSizeCase, 1, ntree);
  free_new_vvector(bootMembershipFlagCase, 1, ntree, NRUTIL_CPTR);
  free_new_vvector(oobMembershipFlagCase, 1, ntree, NRUTIL_CPTR);
  free_uivector(oobSizeCase, 1, ntree);
  free_uivector(ibgSizeCase, 1, ntree);
  free_new_vvector(oobMembershipIndexCase, 1, ntree, NRUTIL_UPTR);
  free_new_vvector(ibgMembershipIndexCase, 1, ntree, NRUTIL_UPTR);
}
