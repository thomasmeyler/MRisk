(self.webpackChunklite=self.webpackChunklite||[]).push([[4129],{99171:(e,t,n)=>{"use strict";n.d(t,{Z:()=>l});var a=n(67294);function i(){return(i=Object.assign||function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var a in n)Object.prototype.hasOwnProperty.call(n,a)&&(e[a]=n[a])}return e}).apply(this,arguments)}var r=a.createElement("path",{fillRule:"evenodd",clipRule:"evenodd",d:"M2.18 2.6a1.82 1.82 0 0 1 3.64 0v2.07H2.18V2.6zm4.37 2.07V2.6C6.55 1.17 5.4 0 4 0a2.58 2.58 0 0 0-2.55 2.6v2.07c-.39 0-.75.17-1.03.46-.27.3-.42.7-.42 1.12v3.17a1.7 1.7 0 0 0 .42 1.12c.14.14.3.26.47.34.18.08.37.12.56.12h5.1c.39 0 .75-.17 1.03-.47.27-.3.42-.7.42-1.11V6.25a1.7 1.7 0 0 0-.42-1.12c-.14-.14-.3-.26-.47-.34a1.35 1.35 0 0 0-.56-.12z",fill:"#757575"});const l=function(e){return a.createElement("svg",i({width:8,height:11,viewBox:"0 0 8 11",fill:"none"},e),r)}},44642:(e,t,n)=>{"use strict";n.d(t,{VK:()=>k,zn:()=>E,kw:()=>b});var a,i=n(59713),r=n.n(i),l=n(63038),o=n.n(l),c=n(67294),s=n(92661),u=n(77280),d=n(92305),m=n(71341);function p(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function v(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?p(Object(n),!0).forEach((function(t){r()(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):p(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}!function(e){e.Delete="delete",e.Reorder="reorder"}(a||(a={}));var f={showReorderView:!1,setShowReorderView:function(){},showBulkDeleteView:!1,setShowBulkDeleteView:function(){},moveOperations:[],deleteOperations:[],addMoveOperation:function(){},addDeleteOperation:function(){},removeDeleteOperation:function(){},resetOperations:function(){},isLoading:!1,setIsLoading:function(){},isBulkActionView:!1},g=(0,c.createContext)(f),k=function(){return(0,c.useContext)(g)},E=function(e){var t=e.children,n=(0,u.Wd)("action"),i=(0,c.useState)(n===a.Reorder),r=o()(i,2),l=r[0],s=r[1],d=(0,c.useState)(n===a.Delete),m=o()(d,2),p=m[0],v=m[1],f=(0,c.useState)([]),k=o()(f,2),E=k[0],b=k[1],h=(0,c.useState)([]),y=o()(h,2),w=y[0],C=y[1],x=(0,c.useState)(!1),S=o()(x,2),O=S[0],I=S[1],D=(0,c.useCallback)((function(e){b(E.concat({move:e}))}),[E]),R=(0,c.useCallback)((function(e){C(w.concat(e))}),[w]),L=(0,c.useCallback)((function(e){var t=w.filter((function(t){return t.entityId!==e.entityId}));C(t)}),[w]),V=(0,c.useCallback)((function(){l?b([]):C([])}),[l]);return c.createElement(g.Provider,{value:{showReorderView:l,setShowReorderView:s,showBulkDeleteView:p,setShowBulkDeleteView:v,moveOperations:E,deleteOperations:w,resetOperations:V,addMoveOperation:D,addDeleteOperation:R,removeDeleteOperation:L,isLoading:O,setIsLoading:I,isBulkActionView:l||p}},t)},b=function(e){var t=e.children,n=e.catalog,i=(0,s.di)("ShowUserList",{username:n.creator.username||"",catalogSlugId:(0,d.EJ)(n)}),r=(0,m.h)({queryParams:{action:a.Reorder}}),l=(0,m.h)({queryParams:{action:a.Delete}}),o=(0,c.useCallback)((function(){r(i)}),[r,i]),u=(0,c.useCallback)((function(){l(i)}),[l,i]);return c.createElement(g.Provider,{value:v(v({},f),{},{setShowReorderView:o,setShowBulkDeleteView:u})},t)}},97996:(e,t,n)=>{"use strict";n.d(t,{G:()=>Z});var a=n(67154),i=n.n(a),r=n(63038),l=n.n(r),o=n(67294),c=n(25267),s=n(38352),u=n(73917),d=n(38356),m=n(42130),p=n(68894),v=n(59713),f=n.n(v),g=n(5977),k=n(44642),E=n(19416),b=n(13791),h=n(77355),y=n(20113),w=n(87691),C=function(e){var t=e.isVisible,n=e.hide,a=e.onConfirm,i=e.loading,r=e.confirmText,l=e.title,c=e.text,s=e.isDestructiveAction;return o.createElement(b.v,{isVisible:t,hide:n,confirmText:o.createElement(E.I,{loading:i,text:r}),isDestructiveAction:s,onConfirm:a,disableConfirm:i,hideOnConfirm:!1},o.createElement(h.x,{paddingBottom:"6px"},o.createElement(y.X6,{scale:"L"},l)),o.createElement(h.x,{paddingBottom:"32px"},o.createElement(w.F,{scale:"L"},c)))};function x(e){var t=e.isVisible,n=e.hide,a=e.deleteCatalog,i=e.loading;return o.createElement(C,{isVisible:t,hide:n,confirmText:"Delete",isDestructiveAction:!0,onConfirm:a,loading:i,title:"Delete list",text:"Deleting this list will remove it from Your library. If others have saved this list, it will also be deleted and removed from their library. Deleting this list will not delete any stories in it."})}var S=n(85208),O=function(e){var t=e.isVisible,n=e.hide,a=e.onConfirm;return o.createElement(C,{isVisible:t,hide:n,loading:!1,text:"If others have saved this list, it will be removed from their library.",confirmText:"Make private",title:"Make list private",onConfirm:a})},I=n(1109),D=n(40201),R=n(93310),L=n(61609),V=n(52069),N=n(75221);function P(e){var t=e.isVisible,n=e.hide,a=e.catalog,i=e.updateCatalog,r=e.loading,c=a.name,s=a.description,u=a.visibility,d=a.id,m=a.type,v=(0,p.O)(!1),f=l()(v,3),g=f[0],k=f[1],y=f[2],C=(0,o.useState)(c),x=l()(C,2),S=x[0],P=x[1],T=(0,o.useState)(!!s),j=l()(T,2),B=j[0],A=j[1],F=(0,o.useState)(s),M=l()(F,2),H=M[0],U=M[1],z=(0,o.useState)(u),_=l()(z,2),G=_[0],Q=_[1],X=a.type===N.HQ.PREDEFINED_LIST,q=(0,o.useCallback)((function(e){P(e.target.value)}),[]),K=(0,o.useCallback)((function(e){U(e.target.value)}),[]),Y=(0,o.useCallback)((function(){Q((function(e){return e===N.n2.PRIVATE?N.n2.PUBLIC:N.n2.PRIVATE}))}),[]),Z=(0,o.useCallback)((function(){var e=S.trim(),t=null==H?void 0:H.trim();i({variables:{catalogId:d,attributes:{type:m,title:e!==c?e:null,description:t!==s?t:null,visibility:G!==u?G:null}}})}),[S,H,G,i,d,m,c,s,u]),J=(0,o.useCallback)((function(){G===N.n2.PRIVATE&&u===N.n2.PUBLIC?k():Z()}),[Z,k,G,u]),W=(0,o.useCallback)((function(){y(),Z()}),[y,Z]),$=S.trim(),ee=(null!==H?H.trim():null)===s&&$===c&&G===u;return g?o.createElement(O,{hide:y,onConfirm:W,isVisible:!0}):o.createElement(b.v,{isVisible:t,hide:n,confirmText:o.createElement(E.I,{loading:r}),isDestructiveAction:!1,onConfirm:J,disableConfirm:!X&&!$||r||ee||(0,I.G6)(S)||(0,I.z6)(H),hideOnConfirm:!1},o.createElement(h.x,{height:"400px"},o.createElement(h.x,{paddingBottom:"60px"},o.createElement(V.Dx,{scale:"L"},"Edit list")),o.createElement(h.x,{textAlign:"left",width:"400px",sm:{width:"100%"}},!X&&o.createElement(h.x,{paddingBottom:"20px"},o.createElement(D.n,{value:S,onChange:q,placeholder:"Give it a name",characterCountLimit:I.lp})),o.createElement(h.x,{paddingBottom:"20px"},B?o.createElement(h.x,{maxHeight:"170px",overflow:"auto"},o.createElement(D.n,{value:null!=H?H:void 0,onChange:K,placeholder:"Description",isMultiline:!0,autoFocus:!0,characterCountLimit:I.B0})):o.createElement(R.r,{onClick:function(){return A(!0)}},o.createElement(w.F,{scale:"L",color:"ACCENT"},"Add a description"))),o.createElement(h.x,null,o.createElement(L.X,{checked:G===N.n2.PRIVATE,onChange:Y,textScale:"L"},"Make it private")))))}var T=n(75675),j=n(70533),B=n(1920),A=n(92661),F=n(78285),M=n(43487),H=n(71341),U=n(50458);function z(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function _(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?z(Object(n),!0).forEach((function(t){f()(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):z(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function G(e){var t=(0,p.O)(!1),n=l()(t,3),a=n[0],i=n[1],r=n[2];return[a,i,(0,o.useCallback)((function(){r(),e()}),[r,e])]}var Q=function(e){var t=e.catalog,n=e.viewer,a=e.ariaExpanded,i=e.ariaId,r=e.hidePopover,c=t.visibility,u=t.id,d=t.name,m=t.description,p=t.type,v=t.postItemsCount,f=t.disallowResponses,E=(0,k.VK)(),b=E.setShowReorderView,h=E.setShowBulkDeleteView,y=(0,M.v9)((function(e){return e.config.authDomain})),C=(0,F.w)(),I=(0,g.TH)(),D=(0,A.$B)(I.pathname),L="user-catalog"===(null==D?void 0:D.route.mediumAppPage),V=(0,H.h)(),z=(0,A.uk)(),Q=(0,A.qt)("ShowYourLists",{}),X=(0,U.yk)(t,y),q=(0,j.sK)(u),K=l()(q,2),Y=K[0],Z=K[1],J=Z.loading,W=Z.error,$=(0,j.qz)(u),ee=l()($,2),te=ee[0],ne=ee[1],ae=ne.loading,ie=ne.error,re=G(r),le=l()(re,3),oe=le[0],ce=le[1],se=le[2],ue=G(r),de=l()(ue,3),me=de[0],pe=de[1],ve=de[2],fe=G(r),ge=l()(fe,3),ke=ge[0],Ee=ge[1],be=ge[2],he=G(r),ye=l()(he,3),we=ye[0],Ce=ye[1],xe=ye[2],Se=(0,S.O2)(t,se,(function(e){window.history.replaceState(null,window.document.title,(0,U.yk)(_(_({},t),{},{name:e}),y))})),Oe=l()(Se,2),Ie=Oe[0],De=Oe[1],Re=De.loading,Le=De.error,Ve=(0,o.useCallback)((function(){L&&(t.creator.username?V(z("ShowUserLists",{username:t.creator.username})):window.location.assign(Q))}),[Q,L,t.creator.username]),Ne=(0,S.h6)({catalog:t,userId:n.id,onCompleted:Ve}),Pe=Ne.deleteCatalog,Te=Ne.loading,je=Ne.error,Be=(0,o.useCallback)((function(){xe(),Re||Ie({variables:{catalogId:u,attributes:{title:d,type:p,visibility:c===N.n2.PRIVATE?N.n2.PUBLIC:N.n2.PRIVATE,description:m}}})}),[u,d,c,Ie,Re,m,p,xe]),Ae=(0,o.useCallback)((function(){c===N.n2.PUBLIC?Ce():Be()}),[Be,c,Ce]),Fe=(0,o.useCallback)((function(){r(),b(!0)}),[r]),Me=(0,o.useCallback)((function(){r(),h(!0)}),[r]);(0,o.useEffect)((function(){(Le||je||W||ie)&&C({toastStyle:"RETRYABLE_ERROR",duration:4e3})}),[Le,je,W,ie,C]);var He=(0,o.useCallback)((function(){J||(Y(),r())}),[J,r]),Ue=(0,o.useCallback)((function(){ae||(te(),r())}),[ae,r]);return o.createElement(o.Fragment,null,X&&o.createElement(B.E,{mediumUrl:X,onClick:r}),o.createElement(s.Sl,null,o.createElement(P,{isVisible:oe,hide:se,catalog:t,updateCatalog:Ie,loading:Re}),o.createElement(R.r,{"aria-controls":i,"aria-expanded":a,onClick:ce},"Edit list info")),v>1&&o.createElement(s.Sl,null,o.createElement(R.r,{"aria-controls":i,"aria-expanded":a,onClick:Me},"Remove items")),o.createElement(s.Sl,null,o.createElement(R.r,{"aria-controls":i,"aria-expanded":a,onClick:Ae},"Make list ",c===N.n2.PUBLIC?"private":"public")),v>1&&o.createElement(s.Sl,null,o.createElement(R.r,{"aria-controls":i,"aria-expanded":a,onClick:Fe},"Reorder items")),o.createElement(s.Sl,null,o.createElement(T.w,{isVisible:ke,hide:be,onConfirm:He,entityTypename:"Catalog"}),o.createElement(R.r,{"aria-controls":i,"aria-expanded":a,onClick:f?Ue:Ee},f?"Show responses":"Hide responses")),t.type!==N.HQ.PREDEFINED_LIST&&o.createElement(s.Sl,null,o.createElement(x,{isVisible:me,hide:ve,deleteCatalog:Pe,loading:Te}),o.createElement(R.r,{"aria-controls":i,"aria-expanded":a,onClick:pe},o.createElement(w.F,{scale:"M",color:"ERROR"},"Delete list"))),o.createElement(O,{isVisible:we,hide:xe,onConfirm:Be}))},X=n(18627),q=n(66411),K=function(e){var t=e.catalog,n=e.ariaExpanded,a=e.hidePopover,i=t.id,r=t.viewerEdge.clapCount,l=(0,S.ok)().clapCatalog,c=(0,S.es)().flagCatalog,u=(0,X.Av)(),d=(0,q.pK)(),m=(0,M.v9)((function(e){return e.config.authDomain})),p=(0,U.yk)(t,m),v=o.useCallback((function(){r&&(l({catalogId:i,numClaps:-r}),u.event("list.clientUnvote",{listId:i,unvoteCount:r,source:d}),a())}),[i,r,d,l,a]),f=o.useCallback((function(){c({catalogId:i}),a()}),[i,c,a]);return o.createElement(o.Fragment,null,o.createElement(s.mX,null,p&&o.createElement(B.E,{mediumUrl:p,onClick:a}),0!==r&&o.createElement(s.Sl,null,o.createElement(R.r,{onClick:v,"aria-expanded":n},"Undo applause for this list")),o.createElement(s.Sl,null,o.createElement(R.r,{onClick:f,"aria-expanded":n},"Report this list"))))},Y=function(e){var t=e.catalog,n=e.viewer,a=e.isResponsive,i=(0,p.O)(!1),r=l()(i,4),c=r[0],v=r[2],f=r[3],g="catalogContentMenu",k=c?"true":"false",E=n.id===t.creator.id,b=o.useCallback((function(){return o.createElement(s.mX,null,E?o.createElement(Q,{catalog:t,ariaExpanded:k,ariaId:g,hidePopover:v,viewer:n}):o.createElement(K,{catalog:t,ariaExpanded:k,hidePopover:v}))}),[E,t,k,v,n]);return o.createElement(u.J,{ariaId:g,isVisible:c,hide:v,popoverRenderFn:b},o.createElement(d.u,{onClick:f,ariaLabel:"More options",icon:o.createElement(m.Z,null),text:a?"More":void 0,tooltipText:"More"}))},Z=function(e){return o.createElement(c.I,null,(function(t){return t?o.createElement(Y,i()({viewer:t},e)):null}))}},24129:(e,t,n)=>{"use strict";n.d(t,{l:()=>N});var a=n(67294),i=n(44642),r=n(97996),l=n(83263),o=n(84739),c=n(1109),s=n(10648),u=n(14818),d=n(77355),m=n(93310),p=n(27323),v=n(30020),f=n(87691),g=n(52069),k=n(18627),E=n(66411),b=n(14646),h=n(18122),y=n(92661),w=n(75221),C=n(99171),x=n(92305),S=n(87498),O=function(e){var t=e.catalog.listItemsConnection.items,n=(0,l.Q)(),i={xs:140,sm:166,md:90,lg:166},r={xs:105,sm:105,md:144,lg:144};return a.useMemo((function(){return a.createElement(l.f,{coverHeight:r[n],coverWidth:i[n],catalogItems:t,isEmbeded:!0})}),[n,t])},I=n(65968),D=n(64238),R=function(e){return{marginBottom:"1px","& path":{fill:e.colorTokens.foreground.neutral.secondary.base}}},L={position:"absolute",inset:"0",border:"none"},V={pointerEvents:"none"},N=function(e){var t=e.catalog,n=e.titleClamp,N=void 0===n?2:n,P=e.maxWidth,T=void 0===P?{xs:"100%",sm:"100%",md:"740px",lg:"740px",xl:"740px"}:P,j=e.marginBottom,B=void 0===j?"40px":j,A=t.name,F=t.visibility,M=t.id,H=t.predefined,U=t.creator,z=(0,c.SZ)(t),_=F!==w.n2.PUBLIC,G=H?(0,x.S6)(H):A,Q=(0,k.Av)(),X=(0,E.pK)(),q=(0,l.Q)(),K=(0,h.g)({onPresentedFn:function(){Q.event("list.presented",{listId:t.id,source:X})}}),Y=(0,b.I)(),Z="sm"===q||"xs"===q,J=(0,y.qt)("ShowUserList",{username:t.creator.username||"",catalogSlugId:(0,x.EJ)(t)}),W=(0,o.B)(U),$=(0,D.o)(U);return a.createElement(d.x,{ref:K,border:"BASE_LIGHTER",borderRadius:"4px",marginBottom:B,maxWidth:T,width:"100%",display:"flex",justifyContent:"space-between",backgroundColor:"BASE_LIGHT",flexDirection:Z?"column":"row",position:"relative"},a.createElement(m.r,{href:J,rules:L,"aria-label":t.name}),a.createElement(d.x,{key:M,display:"flex",flexDirection:"column",padding:"24px 24px 10px",flexGrow:"1",flexShrink:"0",flexBasis:"0",justifyContent:"space-between",wordBreak:"break-word"},a.createElement(p.P,{href:W,"aria-label":U.name||"List creator"},a.createElement(d.x,{display:"flex"},a.createElement(u.z,{miroId:U.imageId||S.gG,alt:U.name||"",diameter:20,freezeGifs:!0}),a.createElement(d.x,{paddingLeft:"8px",zIndex:"1",display:"flex",alignItems:"center"},a.createElement(f.F,{scale:"M",color:"DARKER",clamp:1},U.name),$&&U.id&&a.createElement(I.G,{marginLeft:"2px",marginTop:"2px",size:"S",userId:U.id})))),a.createElement(d.x,{marginTop:"12px"},a.createElement(g.Dx,{clamp:N,scale:{xs:"XS",sm:"XS",md:"S",lg:"S",xl:"S"}},G)),a.createElement(d.x,{display:"flex",alignItems:"center",justifyContent:"space-between"},a.createElement(d.x,{display:"flex",margin:"6px 0"},a.createElement(f.F,{scale:"S",color:"LIGHTER"},z||"No stories"),_&&a.createElement(d.x,{paddingLeft:"8px"},a.createElement(C.Z,{className:Y(R)}))),a.createElement(d.x,{display:"flex",position:"relative"},a.createElement(s.b,{catalog:t}),a.createElement(i.kw,{catalog:t},a.createElement(v._,{tooltipText:"More",targetDistance:10},a.createElement(r.G,{catalog:t})))))),a.createElement("div",{className:Y(V)},a.createElement(O,{catalog:t})))}},70533:(e,t,n)=>{"use strict";n.d(t,{qz:()=>d,sK:()=>u,SM:()=>m,VC:()=>p});var a=n(63038),i=n.n(a),r=n(21919),l=n(75221),o=n(78285),c={kind:"Document",definitions:[{kind:"OperationDefinition",operation:"mutation",name:{kind:"Name",value:"UpdateCatalogLockResponsesMutation"},variableDefinitions:[{kind:"VariableDefinition",variable:{kind:"Variable",name:{kind:"Name",value:"catalogId"}},type:{kind:"NonNullType",type:{kind:"NamedType",name:{kind:"Name",value:"String"}}}},{kind:"VariableDefinition",variable:{kind:"Variable",name:{kind:"Name",value:"attributes"}},type:{kind:"NonNullType",type:{kind:"NamedType",name:{kind:"Name",value:"UpdateCatalogInput"}}}}],selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"updateCatalog"},arguments:[{kind:"Argument",name:{kind:"Name",value:"catalogId"},value:{kind:"Variable",name:{kind:"Name",value:"catalogId"}}},{kind:"Argument",name:{kind:"Name",value:"attributes"},value:{kind:"Variable",name:{kind:"Name",value:"attributes"}}}],selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"__typename"}},{kind:"InlineFragment",typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Catalog"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"responsesLocked"}}]}}]}}]}}]},s={kind:"Document",definitions:[{kind:"OperationDefinition",operation:"mutation",name:{kind:"Name",value:"UpdateCatalogDisallowResponsesMutation"},variableDefinitions:[{kind:"VariableDefinition",variable:{kind:"Variable",name:{kind:"Name",value:"catalogId"}},type:{kind:"NonNullType",type:{kind:"NamedType",name:{kind:"Name",value:"String"}}}},{kind:"VariableDefinition",variable:{kind:"Variable",name:{kind:"Name",value:"attributes"}},type:{kind:"NonNullType",type:{kind:"NamedType",name:{kind:"Name",value:"UpdateCatalogInput"}}}}],selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"updateCatalog"},arguments:[{kind:"Argument",name:{kind:"Name",value:"catalogId"},value:{kind:"Variable",name:{kind:"Name",value:"catalogId"}}},{kind:"Argument",name:{kind:"Name",value:"attributes"},value:{kind:"Variable",name:{kind:"Name",value:"attributes"}}}],selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"__typename"}},{kind:"InlineFragment",typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Catalog"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"disallowResponses"}}]}}]}}]}}]},u=function(e){var t=(0,o.w)();return(0,r.D)(s,{variables:{catalogId:e,attributes:{type:l.HQ.LISTS,disallowResponses:!0}},onCompleted:function(){t({message:"Responses are now hidden for this list."})}})},d=function(e){var t=(0,o.w)();return(0,r.D)(s,{variables:{catalogId:e,attributes:{type:l.HQ.LISTS,disallowResponses:!1}},onCompleted:function(){t({message:"Responses are now shown for this list."})}})},m=function(e,t,n,a){var o=(0,r.D)(c,{variables:{catalogId:a,attributes:{type:l.HQ.LISTS,responsesLocked:!0}},onCompleted:function(){e.event("responses.locked",{catalogId:a,source:t}),n()}});return i()(o,1)[0]},p=function(e,t){var n=(0,r.D)(c,{variables:{catalogId:t,attributes:{type:l.HQ.LISTS,responsesLocked:!1}},onCompleted:function(){e()}});return i()(n,1)[0]}},1920:(e,t,n)=>{"use strict";n.d(t,{_:()=>E,E:()=>b});var a=n(59713),i=n.n(a),r=n(67294),l=n(77355),o=n(93310),c=n(18627),s=n(78285);function u(){return(u=Object.assign||function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var a in n)Object.prototype.hasOwnProperty.call(n,a)&&(e[a]=n[a])}return e}).apply(this,arguments)}var d=r.createElement("path",{fillRule:"evenodd",clipRule:"evenodd",d:"M3.57 14.67c0-.57.13-1.11.38-1.6l.02-.02v-.02l.02-.02c0-.02 0-.02.02-.02.12-.26.3-.52.57-.8L7.78 9v-.02l.01-.02c.44-.41.91-.7 1.44-.85a4.87 4.87 0 0 0-1.19 2.36A5.04 5.04 0 0 0 8 11.6L6.04 13.6c-.19.19-.32.4-.38.65a2 2 0 0 0 0 .9c.08.2.2.4.38.57l1.29 1.31c.27.28.62.42 1.03.42.42 0 .78-.14 1.06-.42l1.23-1.25.79-.78 1.15-1.16c.08-.09.19-.22.28-.4.1-.2.15-.42.15-.67 0-.16-.02-.3-.06-.45l-.02-.02v-.02l-.07-.14s0-.03-.04-.06l-.06-.13-.02-.02c0-.02 0-.03-.02-.05a.6.6 0 0 0-.14-.16l-.48-.5c0-.04.02-.1.04-.15l.06-.12 1.17-1.14.09-.09.56.57c.02.04.08.1.16.18l.05.04.03.06.04.05.03.04.04.06.1.14.02.02c0 .02.01.03.03.04l.1.2v.02c.1.16.2.38.3.68a1 1 0 0 1 .04.25 3.2 3.2 0 0 1 .02 1.33 3.49 3.49 0 0 1-.95 1.87l-.66.67-.97.97-1.56 1.57a3.4 3.4 0 0 1-2.47 1.02c-.97 0-1.8-.34-2.49-1.03l-1.3-1.3a3.55 3.55 0 0 1-1-2.51v-.01h-.02v.02zm5.39-3.43c0-.19.02-.4.07-.63.13-.74.44-1.37.95-1.87l.66-.67.97-.98 1.56-1.56c.68-.69 1.5-1.03 2.47-1.03.97 0 1.8.34 2.48 1.02l1.3 1.32a3.48 3.48 0 0 1 1 2.48c0 .58-.11 1.11-.37 1.6l-.02.02v.02l-.02.04c-.14.27-.35.54-.6.8L16.23 15l-.01.02-.01.02c-.44.42-.92.7-1.43.83a4.55 4.55 0 0 0 1.23-3.52L18 10.38c.18-.21.3-.42.35-.65a2.03 2.03 0 0 0-.01-.9 1.96 1.96 0 0 0-.36-.58l-1.3-1.3a1.49 1.49 0 0 0-1.06-.42c-.42 0-.77.14-1.06.4l-1.2 1.27-.8.8-1.16 1.15c-.08.08-.18.21-.29.4a1.66 1.66 0 0 0-.08 1.12l.02.03v.02l.06.14s.01.03.05.06l.06.13.02.02.01.02.01.02c.05.08.1.13.14.16l.47.5c0 .04-.02.09-.04.15l-.06.12-1.15 1.15-.1.08-.56-.56a2.3 2.3 0 0 0-.18-.19c-.02-.01-.02-.03-.02-.04l-.02-.02a.37.37 0 0 1-.1-.12c-.03-.03-.05-.04-.05-.06l-.1-.15-.02-.02-.02-.04-.08-.17v-.02a5.1 5.1 0 0 1-.28-.69 1.03 1.03 0 0 1-.04-.26c-.06-.23-.1-.46-.1-.7v.01z",fill:"#A8A8A8"});const m=function(e){return r.createElement("svg",u({width:24,height:24,viewBox:"0 0 24 24",fill:"none"},e),d)};var p=n(38352);function v(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function f(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?v(Object(n),!0).forEach((function(t){i()(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):v(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}var g=function(e){return{display:"inline-flex",alignItems:"center",":hover path":{fill:e.colorTokens.foreground.neutral.primary.base}}},k=function(e){var t=e.copyStyle,n=e.children;switch(t){case"INLINE":return r.createElement(l.x,{display:"inline-block"},r.createElement(m,null),r.createElement(l.x,{display:"inline",marginLeft:"8px"},"Copy link"));case"TEXT":return r.createElement(l.x,{display:"inline-block"},n);default:return null}},E=function(e){var t=e.url,n=e.onClick,a=e.reportData,i=void 0===a?{}:a,l=e.source,u=e.copyStyle,d=e.children,m=(0,s.w)(),p=(0,c.Av)();return r.createElement(o.r,{onClick:function(){navigator.clipboard.writeText(t),m({message:"Link copied",toastStyle:"MESSAGE",duration:2e3}),p.event("copyStoryLink.clicked",f(f({},i),{},{source:l})),n&&n()},rules:"INLINE"===u?g:void 0},r.createElement(k,{copyStyle:u},d))},b=function(e){var t=e.mediumUrl,n=e.onClick;return r.createElement(p.Sl,null,r.createElement(E,{url:t,onClick:n,copyStyle:"TEXT"},"Copy link"))}},10648:(e,t,n)=>{"use strict";n.d(t,{b:()=>w});var a=n(67294),i=n(85208),r=n(6443),l=n(26350),o=n(77355),c=n(30020),s=n(38356),u=n(92661),d=n(45550);function m(){return(m=Object.assign||function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var a in n)Object.prototype.hasOwnProperty.call(n,a)&&(e[a]=n[a])}return e}).apply(this,arguments)}var p=a.createElement("path",{d:"M17.5 3.25a.5.5 0 0 1 1 0v2.5H21a.5.5 0 0 1 0 1h-2.5v2.5a.5.5 0 0 1-1 0v-2.5H15a.5.5 0 0 1 0-1h2.5v-2.5zm-11 4.5a1 1 0 0 1 1-1H11a.5.5 0 0 0 0-1H7.5a2 2 0 0 0-2 2v14a.5.5 0 0 0 .8.4l5.7-4.4 5.7 4.4a.5.5 0 0 0 .8-.4v-8.5a.5.5 0 0 0-1 0v7.48l-5.2-4a.5.5 0 0 0-.6 0l-5.2 4V7.75z",fill:"currentColor"}),v=a.createElement("path",{d:"M10.5 2.75h-6a2 2 0 0 0-2 2v11.5",stroke:"currentColor",strokeLinecap:"round"});const f=function(e){return a.createElement("svg",m({width:24,height:24,viewBox:"0 0 24 24",fill:"none"},e),p,v)};var g=n(92305),k=n(78870),E=function(e){return e?"Remove from Your library":"Save list"},b=function(e){var t,n=e.isFollowing,i=e.isResponsive,r=e.onClick,l=e.disabled;return i&&(t=E(n)),a.createElement(s.u,{onClick:r,text:t,icon:n?a.createElement(d.Z,null):a.createElement(f,null),disabled:l,tooltipText:E(n),ariaLabel:E(n)})},h=function(e){var t=e.catalog,n=e.viewerId,r=e.isResponsive,l=(0,i.Yz)(n,t.id),o=l.followCatalog,c=l.loading,s=(0,i.mC)(n,t.id),u=s.unfollowCatalog,d=s.loading,m=t.viewerEdge.isFollowing,p=m?u:o;return a.createElement(b,{onClick:p,disabled:c||d,isResponsive:r,isFollowing:m})},y=function(e){var t=e.catalog,n=e.isResponsive,i=(0,u.qt)("ShowUserList",{username:t.creator.username||"",catalogSlugId:(0,g.EJ)(t)}),r=(0,k.Rk)(i,{save_list:"true"});return a.createElement(l.R,{operation:"register",susiEntry:"follow_list",redirectTo:r},a.createElement(b,{isResponsive:n}))},w=function(e){var t=e.catalog,n=e.marginLeft,i=e.marginRight,l=void 0===i?"8px":i,s=e.isResponsive,u=(0,r.H)(),d=u.value;return u.loading||t.creator.id===(null==d?void 0:d.id)?null:a.createElement(o.x,{marginLeft:n,marginRight:l,flexShrink:"0"},a.createElement(c._,{tooltipText:E(t.viewerEdge.isFollowing),targetDistance:10},d?a.createElement(h,{catalog:t,viewerId:d.id,isResponsive:s}):a.createElement(y,{catalog:t,isResponsive:s})))}}}]);
//# sourceMappingURL=https://stats.medium.build/lite/sourcemaps/4129.54028579.chunk.js.map