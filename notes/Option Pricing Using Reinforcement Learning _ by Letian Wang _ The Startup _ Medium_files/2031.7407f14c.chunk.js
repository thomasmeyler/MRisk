(self.webpackChunklite=self.webpackChunklite||[]).push([[2031],{68455:(e,t,n)=>{"use strict";n.d(t,{Z:()=>l});var r=n(67294);function a(){return(a=Object.assign||function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e}).apply(this,arguments)}var o=r.createElement("path",{d:"M18.5 4.43a6.9 6.9 0 0 1-2.18.88 3.45 3.45 0 0 0-2.55-1.12 3.49 3.49 0 0 0-3.49 3.48c0 .28.03.55.07.81a9.91 9.91 0 0 1-7.17-3.67 3.9 3.9 0 0 0-.5 1.74 3.6 3.6 0 0 0 1.56 2.92 3.36 3.36 0 0 1-1.55-.44.15.15 0 0 0 0 .06c0 1.67 1.2 3.08 2.8 3.42-.3.06-.6.1-.94.12l-.62-.06A3.5 3.5 0 0 0 7.17 15a7.33 7.33 0 0 1-4.36 1.49L2 16.44A9.96 9.96 0 0 0 7.36 18c6.4 0 9.91-5.32 9.9-9.9v-.5A6.55 6.55 0 0 0 19 5.79a6.18 6.18 0 0 1-2 .56 3.33 3.33 0 0 0 1.5-1.93"});const l=function(e){return r.createElement("svg",a({width:21,height:21,viewBox:"0 0 21 21"},e),o)}},72469:(e,t,n)=>{"use strict";n.d(t,{Sk:()=>i,$:()=>c});var r=n(67294),a=n(77355),o=n(29746),l=n(14646),i=function(){return{overflowX:"scroll","::-webkit-scrollbar":{display:"none"},"scrollbar-width":"none","-ms-overflow-style":"none"}},c=function(e){var t=e.children,n=e.height,c=void 0===n?40:n,s=(0,l.I)();return r.createElement(a.x,{position:"relative",width:"100%",overflow:"hidden",height:(0,o.a)(c)},r.createElement(a.x,{position:"absolute",top:"0",width:"100%"},r.createElement("div",{className:s([i,function(){return{height:(0,o.a)(c+20)}}])},t)))}},47834:(e,t,n)=>{"use strict";n.d(t,{s:()=>I,j:()=>k});var r=n(45578),a=n.n(r),o=n(98913),l=n.n(o),i=n(67294),c=n(14818),s=n(28695),u=n(77355),d=n(27323),m=n(69992),p=n(30020),V=n(14646),f=n(87498),v=n(45932),h=n(18978),g=n(68427),E=n(84739),x=function(e){var t=e.diameter,n=e.zIndex,r=e.showBorder,a=e.borderColor;return function(e){return{display:"block",width:"".concat(r?t+4:t,"px"),height:"".concat(r?t+4:t,"px"),borderRadius:"50%",border:r?"".concat(2,"px solid ").concat(null!=a?a:e.backgroundColor):"none",zIndex:n}}},y=function(e){return{background:e.colorTokens.background.neutral.secondary.base,border:"2px solid white",borderRadius:"50%",width:"36px",height:"36px"}},b=function(e,t){return{display:"grid",alignItems:"end",gridTemplateColumns:"repeat(".concat(t,", ").concat(e.toPrecision(3),"%)")}},w=function(e){var t=e.numUsers,n=e.diameter,r=e.withAnimation,a=e.children,o=(0,V.I)(),l=(0,v.P)(),c=n*t,s=100/(t+1),d=t>1?(t>1?(n+4)*t:n)*(1-s/100):n,m=r?[b(s,t),function(){return l}]:b(s,t);return i.createElement(u.x,{width:"".concat(d,"px")},i.createElement(u.x,{width:"".concat(c,"px")},i.createElement("div",{className:o(m)},a)))},I=function(e){var t=e.maxNumEntities,n=void 0===t?12:t,r=e.entities,o=e.error,l=e.loading,c=e.entityElementKey,s=e.showNextAvatarOnTop,u=void 0!==s&&s,d=e.diameter,m=void 0===d?36:d,p=e.noBackgroundShadow,V=Math.min(r.length,n),f=i.useMemo((function(){return a()(r,(function(e){return e.id})).slice(0,V)}),[r,V]);return(V=f.length)?o?null:l?i.createElement(S,{numUsers:V,diamereter:m,entityElementKey:c,withAnimation:!0}):i.createElement(w,{numUsers:V,diameter:m},f.map((function(e,t){var n,a,o=u?t:r.length-t;if("Collection"===e.__typename)n=null===(a=e.avatar)||void 0===a?void 0:a.id;else if("User"===e.__typename)n=e.imageId;else if("NewsletterV3"===e.__typename){var l,s,d;n=e.collection?e.avatarImageId||(null===(l=e.collection)||void 0===l||null===(s=l.avatar)||void 0===s?void 0:s.id):null===(d=e.user)||void 0===d?void 0:d.imageId}return i.createElement(A,{key:"".concat(c,"-").concat(e.id),diameter:m,zIndex:o,showBorder:f.length>1,miroId:n,entityName:e.name,noBackgroundShadow:p})}))):null},A=function(e){var t=e.diameter,n=e.zIndex,r=e.showBorder,a=e.miroId,o=e.entityName,l=e.noBackgroundShadow,s=void 0===l||l,u=e.showAvatarBorder,d=void 0!==u&&u,m=e.showHoverState,p=void 0!==m&&m,v=e.borderColor,h=(0,V.I)();return i.createElement("div",{className:h(x({diameter:t,zIndex:n,showBorder:r,borderColor:v}))},i.createElement(c.z,{miroId:a||f.gG,alt:o||"",diameter:t,freezeGifs:!0,noBackgroundShadow:s,showBorder:d,showHoverState:p}))},S=function(e){var t=e.numUsers,n=e.diamereter,r=e.withAnimation,a=e.entityElementKey,o=(0,V.I)();return i.createElement(w,{numUsers:t,diameter:n,withAnimation:r},l()(t,(function(e){return i.createElement(u.x,{key:"".concat(a,"-placeholder-").concat(e),zIndex:t-e},i.createElement("div",{className:o(y)}))})))},k=function(e){var t,n=e.author,r=e.collection,a=e.authorSize,o=e.collectionImageSize,l=e.collectionOffset,c=e.borderColor,V=e.withAuthorTooltip,f=null!=a?a:44,v=null!=o?o:24,x=(0,g.B)(),y=(0,E.I)();if(!n)return null;var b=i.createElement(A,{diameter:f,zIndex:0,showBorder:!0,miroId:n.imageId,entityName:n.name,noBackgroundShadow:!0,showAvatarBorder:!0,showHoverState:!0,borderColor:c});return i.createElement(u.x,{display:"flex",alignItems:"baseline"},i.createElement(d.P,{href:y(n)},V?i.createElement(m.$,{placement:"bottom",targetDistance:10,mouseLeaveDelay:100,mouseEnterDelay:p.w,popoverRenderFn:function(){return function(e){return i.createElement(s.K,{user:e})}(n)},role:"tooltip"},b):b),r&&i.createElement(d.P,{href:x(r)},i.createElement(u.x,{display:"flex",marginLeft:null!=l?l:"-12px",position:"relative"},i.createElement(m.$,{placement:"bottom",targetDistance:10,mouseLeaveDelay:100,mouseEnterDelay:p.w,popoverRenderFn:function(){return function(e){return i.createElement(h.L,{collection:e})}(r)},role:"tooltip"},i.createElement(A,{diameter:v,zIndex:1,showBorder:!0,miroId:null===(t=r.avatar)||void 0===t?void 0:t.id,entityName:r.name,noBackgroundShadow:!0,showAvatarBorder:!0,showHoverState:!0,borderColor:c})))))}},95634:(e,t,n)=>{"use strict";n.d(t,{I:()=>c});var r=n(67294),a=n(77355),o=n(87691),l=n(14646),i=function(e){return function(t){return{position:"absolute",clip:"rect(0px 14px 14px -3px)",":after":{content:"''",display:"block",width:"11px",height:"11px",background:null!=e?e:t.colorTokens.background.neutral.secondary.base,borderBottomRightRadius:"1px",transform:"rotate(45deg) translate(-4px, -4px)"}}}},c=function(e){var t=e.children,n=e.padding,c=void 0===n?{xs:"16px",sm:"16px",md:"16px",lg:"16px 24px",xl:"16px 24px"}:n,s=e.borderRadius,u=void 0===s?"8px":s,d=e.backgroundColor,m=e.shouldShowArrowDown,p=void 0!==m&&m,V=(0,l.I)();return r.createElement(r.Fragment,null,r.createElement("div",{className:V((function(e){return{position:"relative",padding:c,borderRadius:u,background:null!=d?d:e.colorTokens.background.neutral.secondary.base}}))},r.createElement(o.F,{scale:"M",color:"DARKER"},t)),p&&r.createElement(a.x,{marginLeft:{xs:"18px",sm:"18px",md:"26px",lg:"26px",xl:"26px"}},r.createElement("div",{className:V(i(d))})))}},2031:(e,t,n)=>{"use strict";n.d(t,{D:()=>Ne});var r=n(67294),a=n(33380),o=n(23450),l=n.n(o),i=n(95634),c=n(25735),s=n(6443),u=n(26350),d=n(28451),m=n(43822),p=n(18627),V=n(66411),f=n(14646),v=n(39944),h=n(92661),g=n(43487),E=function(e){return{fontWeight:e.newFonts.detail.boldWeight}},x=function(e){switch(e){case 0:return"metered_view_3";case 1:return"metered_view_2";case 2:default:return"smart_meter"}},y=function(e){var t=e.children,n=e.isLoggedIn,a=e.post,o=e.numFreeStoriesRemaining,l=(0,h.H2)(),i=a.id;return n?r.createElement(v.M,{source:"upgrade_membership",dimension:"post_counter",sourceProviderData:{postId:i},eventData:{postId:i}},r.createElement(m.a,{post:a,redirectUrl:l("ShowPay",{})},t)):r.createElement(V.cW,{source:{postId:a.id}},r.createElement(u.R,{operation:"register",susiEntry:x(o)},t))},b=function(e){var t=e.post,n=(0,f.I)();return r.createElement(y,{isLoggedIn:!0,post:t,numFreeStoriesRemaining:0},r.createElement(i.I,{padding:{xs:"16px",sm:"16px",md:"16px",lg:"16px 24px",xl:"16px 24px"},shouldShowArrowDown:!0},r.createElement("strong",{className:n(E)},"This member-only story is on us.")," ",r.createElement(r.Fragment,null,r.createElement("span",{className:n({textDecoration:"underline"})},"Upgrade")," to access all of Medium.")))},w=function(e){var t=e.post,n=e.meteringInfo,a=(0,f.I)(),o=(0,p.Av)(),u=(0,g.v9)((function(e){return e.config.productName})),m=(0,s.H)(),V=m.loading,v=m.error,h=!!m.value,x=(0,c.d)("enable_maim_the_meter"),w=x.value,I=x.loading,A=null!=n&&n.postIds?n.postIds.length:0,S=null!=n&&n.unlocksRemaining?n.unlocksRemaining:0;return r.useEffect((function(){V||v||o.event("meter.viewed",{uiType:h?d.j.LIHighlightCTA:d.j.LOHighlightCTA,postId:t.id,meterCount:A})}),[V,v,h]),V||v||I?null:h&&w?r.createElement(b,{post:t}):r.createElement(y,{isLoggedIn:h,post:t,numFreeStoriesRemaining:S},r.createElement(i.I,{padding:{xs:"16px",sm:"16px",md:"16px",lg:"16px 24px",xl:"16px 24px"},shouldShowArrowDown:!0},0===S?r.createElement(r.Fragment,null,"This is your ",r.createElement("strong",{className:a(E)},"last free member-only story")," this month."):r.createElement(r.Fragment,null,"You have"," ",r.createElement("strong",{className:a(E)},S," free member-only"," ",l()("story",S)," left")," ","this month.")," ",h?r.createElement(r.Fragment,null,r.createElement("span",{className:a({textDecoration:"underline"})},"Upgrade")," for unlimited access."):r.createElement(r.Fragment,null,r.createElement("span",{className:a({textDecoration:"underline"})},"Sign up")," for ",u," and get an extra one.")))},I=n(63038),A=n.n(I),S=n(27517),k=n(51098),T=n(6402),R=n(54712),O=n(98067),L=n(35989),B=n(47834),N=n(61796),_=n(96370),P=n(77355),C=n(21372),D=n(32342),F=n(91743),M=n(84739),z=n(65968),j=n(64238),H=n(26700),U=n(17311),K=n(34796),W=n(84663),G=n(56804),Z=n(64718),Q={kind:"Document",definitions:[{kind:"OperationDefinition",operation:"query",name:{kind:"Name",value:"MaybeTextToSpeechQuery"},variableDefinitions:[{kind:"VariableDefinition",variable:{kind:"Variable",name:{kind:"Name",value:"postId"}},type:{kind:"NonNullType",type:{kind:"NamedType",name:{kind:"Name",value:"ID"}}}}],selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"post"},arguments:[{kind:"Argument",name:{kind:"Name",value:"id"},value:{kind:"Variable",name:{kind:"Name",value:"postId"}}}],selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"detectedLanguage"}},{kind:"Field",name:{kind:"Name",value:"wordCount"}}]}}]}}]},Y=n(68356),X=n.n(Y),$=n(10374),q=n(50742),J=n(55641),ee=n(78870),te=function(e){var t=e.postId,n=(0,s.H)().value,a="post_audio_button",o=(0,ee.Rk)((0,h.qt)("ShowPay",{}),{dimension:a,postId:t});return n?r.createElement(v.M,{source:"upgrade_membership",dimension:a},r.createElement(m.a,{redirectUrl:o},r.createElement(J.Z,null))):r.createElement(u.R,{operation:"register",susiEntry:a,actionUrl:o},r.createElement(J.Z,null))},ne=X()({loader:function(){return n.e(6391).then(n.bind(n,48574))},modules:["src/components/post/text-to-speech/SpeechifyWidget"],webpack:function(){return[48574]},loading:function(e){var t=e.children;return r.createElement(r.Fragment,null,t)},render:(0,q.n)("SpeechifyWidget")}),re=r.forwardRef((function(e,t){var n=e.postId,a=e.postBodyRef,o=e.isLockedPreviewOnly,l=(0,p.Av)(),i=(0,s.H)().value,u=(0,c.V)({name:"enable_members_only_audio",placeholder:!1}),d=!(null==i||!i.mediumMemberAt),m=(0,$.Ij)().paragraphRefsMappers.paragraphTitlesRefsByStyleMap,V=(0,r.useState)(!0),f=A()(V,2),v=f[0],h=f[1];r.useEffect((function(){return l.event("experiment.eligible",{experimentId:"ec7b685b3f40"}),h(!0)}),[m]);var g=!d&&(o||u);return r.createElement(P.x,{display:"inline-flex",alignItems:"flex-start",boxSizing:"border-box"},r.createElement(P.x,{flexGrow:"1"},r.createElement(_.P,{size:"full"},r.createElement(P.x,{ref:t,display:"flex"},g?u?r.createElement(te,{postId:n}):null:v?r.createElement(J.O,{onClick:function(){if(v){var e=new Audio;e.autoplay=!0,e.src="data:audio/mpeg;base64,SUQzBAAAAAABEVRYWFgAAAAtAAADY29tbWVudABCaWdTb3VuZEJhbmsuY29tIC8gTGFTb25vdGhlcXVlLm9yZwBURU5DAAAAHQAAA1N3aXRjaCBQbHVzIMKpIE5DSCBTb2Z0d2FyZQBUSVQyAAAABgAAAzIyMzUAVFNTRQAAAA8AAANMYXZmNTcuODMuMTAwAAAAAAAAAAAAAAD/80DEAAAAA0gAAAAATEFNRTMuMTAwVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVf/zQsRbAAADSAAAAABVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVf/zQMSkAAADSAAAAABVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV",h(!1)}}}):r.createElement(ne,{postId:n,postBodyRef:a},r.createElement(J.Z,{isPlaying:!0}))))))})),ae=function(e){var t,n,a,o,l=e.postId,i=e.postBodyRef,c=e.isLockedPreviewOnly,s=e.isPublished,u=(0,Z.a)(Q,{variables:{postId:l}}),d=u.data,m=u.error,p=u.loading;if(!l||p||m)return null;var V=null!==(t=null==d||null===(n=d.post)||void 0===n?void 0:n.wordCount)&&void 0!==t?t:0,f=null!==(a=null==d||null===(o=d.post)||void 0===o?void 0:o.detectedLanguage)&&void 0!==a?a:"";return!s||s&&"en"===f&&V>20?r.createElement(re,{postId:l,postBodyRef:i,isLockedPreviewOnly:c}):null},oe=n(75761),le=n(35473),ie=n(73279),ce=n(32317),se=n(28695),ue=n(93310),de=n(69992),me=n(30020),pe=n(18634),Ve=n(21755),fe=n(87691),ve=n(67713),he=n(17583),ge=n(58992),Ee=n(19918),xe=n(1444),ye=n(21232),be=n(93589),we=function(e){var t,n=e.avatars,a=void 0===n?null:n,o=e.publishedAt,l=e.timeToRead,i=e.post,c=e.publisher,s=e.postBodyRef,u=e.isLockedPreviewOnly,d=(0,r.useContext)(ye.f).openSidebar;return r.createElement(P.x,{width:"100%",speechifyIgnore:!0},r.createElement(le.Y,{avatar:a,title:r.createElement(Ie,{author:c}),description:r.createElement(Ae,{publishedAt:o,timeToRead:l,post:i}),leftMargin:"12px",alignItems:{xs:"flex-start",sm:"flex-start",md:"center",lg:"center",xl:"center"}}),r.createElement(Ee.T,{leftButtons:r.createElement(r.Fragment,null,r.createElement(P.x,{width:"74px"},r.createElement(G.S,{post:i,buttonStyle:"SUBTLE_MARGIN",hasDialog:!0,shouldShowResponsiveLabelText:!0,shouldHideClapsText:!0,susiEntry:"clap_footer",buttonColor:"LIGHTER",countScale:"S"})),r.createElement(oe.h,{trackingData:{postId:i.id},responsesCount:null===(t=i.postResponses)||void 0===t?void 0:t.count,allowResponses:i.allowResponses,isLimitedState:i.isLimitedState,handleClick:d,iconStylesOverride:{marginTop:"0px"},countStylesOverride:{marginLeft:"4px",marginTop:"0px"},disabledTooltipText:"Responses hidden",responsesCountScale:"S"})),rightButtons:r.createElement(r.Fragment,null,r.createElement(pe.y,{md:!0,lg:!0,xl:!0},r.createElement(xe.o,{post:i,susiEntry:"bookmark_footer",buttonStyle:"ICON_SUBTLE"})),r.createElement(ae,{postId:i.id,postBodyRef:s,isLockedPreviewOnly:u,isPublished:!!i.isPublished}),r.createElement(W.I,{post:i,source:"post_actions_header",isResponsive:!0}),r.createElement(F.t,{post:i,isResponsive:!0}))}))},Ie=function(e){var t=e.author,n=(0,he.s)(),a=(0,M.B)(t),o=(0,j.o)(t),l=n===Ve.j.xs,i=(0,s.H)().value,c=i&&i.id===t.id;return t&&t.name?r.createElement(P.x,{display:"flex",alignItems:"center",marginBottom:"2px"},r.createElement(P.x,{display:"flex",flexWrap:"nowrap",alignItems:"center"},r.createElement(P.x,{display:"flex",alignItems:"center"},l?r.createElement(fe.F,{color:"DARKER",scale:"L"},r.createElement(ue.r,{linkStyle:"SUBTLE",inline:!0,href:a},t.name)):r.createElement(de.$,{placement:"bottom",targetDistance:10,mouseLeaveDelay:100,mouseEnterDelay:me.w,popoverRenderFn:function(){return r.createElement(se.K,{user:t})}},r.createElement(fe.F,{color:"DARKER",scale:"L"},r.createElement(ue.r,{linkStyle:"SUBTLE",inline:!0,href:a},t.name)))),o&&r.createElement(P.x,{marginLeft:"2px",marginTop:"1px"},r.createElement(z._,{size:"S"})),!c&&r.createElement(r.Fragment,null,r.createElement(ie.O,{display:"inline",margin:"0 8px"}),r.createElement(fe.F,{scale:"L"},r.createElement(ce.B,{user:t,susiEntry:"post_header",isLinkStyle:!0}))))):null},Ae=function(e){var t=e.publishedAt,n=e.timeToRead,a=e.post,o=a.collection;return r.createElement(P.x,{display:"flex",alignItems:"flex-start",flexWrap:"wrap",flexDirection:{xs:"column",sm:"column",md:void 0,lg:void 0,xl:void 0}},!!o&&r.createElement(P.x,{display:"flex",marginBottom:{xs:"2px",sm:"2px",md:void 0,lg:void 0,xl:void 0}},r.createElement(Se,{collection:o,post:a}),r.createElement(pe.y,{md:!0,lg:!0,xl:!0},r.createElement(ie.O,{display:"inline",margin:"0 8px"}))),r.createElement(fe.F,{color:"LIGHTER",scale:"M",tag:"span"},r.createElement(be.Q,{middotPadding:"8px",flexGrow:"1",display:"flex"},n||null,t?r.createElement(U.h,{timestamp:t}):r.createElement(K.F,{post:a}))))},Se=function(e){var t=e.collection,n=e.post,a=(0,ge.l)(n),o=(0,ve.n)({name:"detail",scale:"M",color:"LIGHTER"}),l=(0,f.I)();return a?r.createElement("div",{className:l([o,{display:"flex",whiteSpace:"pre-wrap"}])},r.createElement(P.x,{flexShrink:"0",tag:"span",marginRight:"4px"},"Published in"),r.createElement(H.q,{collection:t,clamp:1})):null},ke=function(e){var t=e.meteringInfo,n=e.post,a=e.hasBannerImage,o=function(e){var t,n=e.post,a=e.meteringInfo,o=(0,R.xg)(),l=(0,r.useState)(!1),i=A()(l,2),u=i[0],d=i[1],m=(0,S.I0)();(0,r.useEffect)((function(){d(!0),m((0,O.Dl)(!1))}),[]);var p=(0,s.H)(),V=p.loading,f=p.error,v=p.value,h=(0,k.a)(),g=h.loading,E=h.shouldShowIncognitoRegwall,x=(0,c.d)("enable_maim_the_meter"),y=x.value,b=x.error,w=x.loading;if((0,r.useEffect)((function(){b&&T.k.error({error:b},"Error fetching enable_maim_the_meter flag")}),[b]),w)return!1;if(f||V)return!1;var I=!(null==v||!v.id||null===(t=n.creator)||void 0===t||!t.id||v.id!==n.creator.id),L=!(null==v||!v.mediumMemberAt);return!y||L||I||n.content.validatedShareKey?!(!a||o&&!u||g||E||!n.isLocked||"LOCKED_POST_SOURCE_SYNDICATED"===n.lockedSource||L||I||!a.postIds.includes(n.id)):n.isLocked&&!n.content.isLockedPreviewOnly}({post:n,meteringInfo:t});return r.createElement(r.Fragment,null,r.createElement(P.x,{marginTop:o&&!a?{xs:"8px",sm:"8px",md:"24px",lg:"24px",xl:"24px"}:{xs:"32px",sm:"32px",md:"40px",lg:"40px",xl:"40px"}}),o&&r.createElement(P.x,{display:"flex",justifyContent:"center"},r.createElement(P.x,{width:"100%",minWidth:"0",maxWidth:"728px",marginBottom:"24px",marginRight:a?void 0:"8px",marginLeft:a?void 0:"8px"},r.createElement(w,{post:n,meteringInfo:t}))))},Te=function(e){var t=e.post,n=e.postBodyRef,o=(0,f.I)(),l=null==t?void 0:t.creator,i=null==t?void 0:t.collection;return l?r.createElement(a.TA,{className:o({xs:"8px",sm:"8px",md:"16px",lg:"16px",xl:"16px"}),name:"byline",type:"byline",offset:{left:-600}},r.createElement(P.x,{display:"flex",justifyContent:"space-between",speechifyIgnore:!0},r.createElement(we,{avatars:r.createElement(B.j,{author:l,collection:i,withAuthorTooltip:!0}),publishedAt:t.firstPublishedAt,publisher:l,timeToRead:!t.isShortform&&t.readingTime?"".concat((0,C.Vd)(t.readingTime)," min read"):void 0,post:t,postBodyRef:n,isLockedPreviewOnly:!!t.content.isLockedPreviewOnly}))):null};function Re(e,t){return"".concat(t,"_").concat(e[t]?e[t].length:0)}function Oe(e,t,n,r,a){e[t]||(e[t]=[]),e[t].push({order:n,component:r,insertType:a})}function Le(e,t,n,a){return function(o){if(!e.isLocked)return o;var l=a.kickerIndex||a.titleIndex,i=a.bannerImageIndex,c="number"==typeof a.titleIndex,s="number"==typeof i,u=r.createElement(P.x,{marginBottom:c?"16px":"0px"},r.createElement(L.U,{post:e,label:"Member-only story",showLabelMobile:!0}));return s&&c&&i<l?Oe(o,n[l]&&n[l].name,"before",r.createElement(P.x,{key:"meter",speechifyIgnore:!0},r.createElement(ke,{post:e,meteringInfo:t,hasBannerImage:!0}),u),"METER"):Oe(o,"first","before",r.createElement(P.x,{key:"meter",speechifyIgnore:!0},r.createElement(ke,{post:e,meteringInfo:t}),r.createElement(_.P,{size:"app"},u)),"METER"),o}}function Be(e,t,n,a){return function(o){var l="number"==typeof n.titleIndex,i=n.subtitleIndex||n.titleIndex||0,c=t[i]&&t[i].name;if(c&&0===i&&!l){var s=r.createElement(P.x,{marginTop:{xs:"24px",sm:"24px",md:"32px",lg:"32px",xl:"32px"},speechifyIgnore:!0},r.createElement(Te,{post:e,postBodyRef:a,key:"insert_postBylineGroupComponent_".concat(Re(o,"first"))}));Oe(o,"first","before",r.createElement(_.P,{size:"app",key:"insert_MaxWidth_PostBylineGroupComponent_".concat(Re(o,"first"))},s),"BYLINE")}else c&&Oe(o,c,"after",r.createElement(Te,{post:e,postBodyRef:a,key:"insert_PostBylineGroupComponent_".concat(Re(o,c))}),"BYLINE");return o}}function Ne(e,t,n){var r,a,o=e&&e.content&&e.content.bodyModel&&e.content.bodyModel.paragraphs||void 0,l=(null==e||null===(r=e.content)||void 0===r||null===(a=r.bodyModel)||void 0===a?void 0:a.sections)||[],i=(0,D.wg)(l,0,null==o?void 0:o.length),c=o.filter((function(e,t){return t<i}));if(o){var s=(0,N.L)(c);return[Le(e,n,o,s),Be(e,o,s,t)].reduce((function(e,t){return t(e)}),{})}}},1444:(e,t,n)=>{"use strict";n.d(t,{o:()=>s});var r=n(67294),a=n(70929),o=n(6443),l=n(75221),i=n(43487),c=n(50458),s=function(e){var t=e.post,n=e.susiEntry,s=e.buttonStyle,u=t.id,d=t.visibility,m=(0,i.v9)((function(e){return e.config.authDomain}));return(0,o.H)().loading||d===l.Wn.UNLISTED?null:r.createElement(a.o,{kind:l.ej.POST,target:t,buttonStyle:s,susiEntry:n,susiActionUrl:(0,c.XE)(m,u)})}},84663:(e,t,n)=>{"use strict";n.d(t,{I:()=>s});var r=n(67294),a=n(25145),o=n(1920),l=n(85805),i=n(37597),c=n(38352),s=function(e){var t=e.post,n=e.source,s=e.isResponsive,u=t.title,d=t.id,m=t.isPublished,p=(0,a.D)()(t);return p||m?r.createElement(l.A,{ariaId:"postFooterSocialMenu",source:{name:n},url:p,title:u,ariaLabel:"Share Post",postId:d,isResponsive:s},(function(e){return r.createElement(r.Fragment,null,p&&r.createElement(r.Fragment,null,r.createElement(c.Sl,null,r.createElement(o._,{url:p,onClick:e,reportData:{postId:t.id},source:n,copyStyle:"INLINE"}))),m&&r.createElement(r.Fragment,null,r.createElement(c.oK,{paddingTopBottom:"5px"}),r.createElement(c.Sl,{paddingTopBottom:"5px"},r.createElement(i.f,{socialPlatform:"TWITTER",buttonStyle:"LINK_INLINE_SHORT_LABEL",postId:t.id})),r.createElement(c.Sl,{paddingTopBottom:"5px"},r.createElement(i.f,{socialPlatform:"FACEBOOK",buttonStyle:"LINK_INLINE_SHORT_LABEL",postId:t.id})),r.createElement(c.Sl,{paddingTopBottom:"5px"},r.createElement(i.f,{socialPlatform:"LINKEDIN",buttonStyle:"LINK_INLINE_SHORT_LABEL",postId:t.id}))))})):null}},55641:(e,t,n)=>{"use strict";n.d(t,{O:()=>g,Z:()=>h});var r=n(63038),a=n.n(r),o=n(67294),l=n(17417),i=n(38356),c=n(66411),s=n(77280);function u(){return(u=Object.assign||function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e}).apply(this,arguments)}var d=o.createElement("path",{fillRule:"evenodd",clipRule:"evenodd",d:"M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM2 12a10 10 0 1 1 20 0 10 10 0 0 1-20 0zm7.25-3c0-.28.22-.5.5-.5h.5c.28 0 .5.22.5.5v6a.5.5 0 0 1-.5.5h-.5a.5.5 0 0 1-.5-.5V9zm4.5-.5a.5.5 0 0 0-.5.5v6c0 .28.22.5.5.5h.5a.5.5 0 0 0 .5-.5V9a.5.5 0 0 0-.5-.5h-.5z",fill:"currentColor"});const m=function(e){return o.createElement("svg",u({width:24,height:24,viewBox:"0 0 24 24",fill:"none"},e),d)};function p(){return(p=Object.assign||function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e}).apply(this,arguments)}var V=o.createElement("path",{fillRule:"evenodd",clipRule:"evenodd",d:"M3 12a9 9 0 1 1 18 0 9 9 0 0 1-18 0zm9-10a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm3.38 10.42l-4.6 3.06a.5.5 0 0 1-.78-.41V8.93c0-.4.45-.63.78-.41l4.6 3.06c.3.2.3.64 0 .84z",fill:"currentColor"});const f=function(e){return o.createElement("svg",p({width:24,height:24,viewBox:"0 0 24 24",fill:"none"},e),V)};var v=n(68894),h=function(e){var t=e.onClick,n=e.isPlaying,r=n?"Pause":"Listen";return o.createElement(i.u,{icon:n?o.createElement(m,null):o.createElement(f,null),onClick:t,text:r,tooltipText:r,ariaLabel:r})},g=function(e){var t=e.onClick,n=(0,s.PM)(),r=(0,c.P7)(n||"").susiEntry,i=(0,v.O)("post_audio_button"===r),u=a()(i,3),d=u[0],m=u[2];return o.createElement(l.A,{isVisible:d,message:"You can now listen to stories",dismissText:"Got it",dismiss:m,targetDistance:10},o.createElement(h,{onClick:t}))}},19918:(e,t,n)=>{"use strict";n.d(t,{T:()=>d});var r=n(67294),a=n(72469),o=n(18634),l=n(14646),i={display:"flex",alignItems:"center","> *":{marginRight:{xs:"8px",sm:"8px",md:"24px",lg:"24px",xl:"24px"},flexShrink:0},"> :last-child":{marginRight:{xs:"24px",sm:"8px",md:0,lg:0,xl:0}}},c=function(e){return"solid 1px ".concat(e.colorTokens.background.neutral.secondary.base)},s=function(e){return{xs:void 0,sm:void 0,md:c(e),lg:c(e),xl:c(e)}},u=function(e){return function(t){return{display:"flex",justifyContent:"space-between",borderTop:s(t),borderBottom:s(t),margin:{xs:"".concat(null!=e?e:"24px"," -24px 0"),sm:"".concat(null!=e?e:"24px"," 0 0"),md:"".concat(null!=e?e:"32px"," 0 0"),lg:"".concat(null!=e?e:"32px"," 0 0"),xl:"".concat(null!=e?e:"32px"," 0 0")},padding:{xs:"0",sm:"0",md:"3px 8px",lg:"3px 8px",xl:"3px 8px"}}}},d=function(e){var t=e.leftButtons,n=e.rightButtons,c=e.marginTop,s=(0,l.I)();return r.createElement("div",{className:s(u(c))},r.createElement(o.y,{md:!0,lg:!0,xl:!0,displayValue:"flex",alignItems:"center"},t),r.createElement("div",{className:s([i,a.Sk])},r.createElement(o.y,{xs:!0,width:"16px"}),n))}},51098:(e,t,n)=>{"use strict";n.d(t,{a:()=>m});var r=n(87757),a=n.n(r),o=n(48926),l=n.n(o),i=n(63038),c=n.n(i),s=n(73145),u=n(67294),d=n(6443),m=function(){var e=(0,d.H)(),t=e.value,n=e.loading,r=!!t,o=(0,u.useState)(),i=c()(o,2),m=i[0],p=i[1],V=(0,u.useCallback)(l()(a().mark((function e(){var t,n;return a().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.prev=0,e.next=3,(0,s.r)();case 3:t=e.sent,n="Unknown"!==t.browserName&&t.isPrivate,p(n),e.next=11;break;case 8:e.prev=8,e.t0=e.catch(0),p(!1);case 11:case"end":return e.stop()}}),e,null,[[0,8]])}))),[]);return(0,u.useEffect)((function(){void 0===m&&V()}),[m]),{loading:void 0===m||!!n,shouldShowIncognitoRegwall:!r&&!!m}}},99950:(e,t,n)=>{"use strict";n.d(t,{T:()=>F});var r=n(67154),a=n.n(r),o=n(67294),l=n(14646);function i(){return(i=Object.assign||function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e}).apply(this,arguments)}var c=o.createElement("path",{d:"M18.26 10.55c0-4.3-3.47-7.79-7.75-7.79a7.77 7.77 0 0 0-7.75 7.79 7.77 7.77 0 0 0 6.54 7.68v-5.49H7.4v-2.2h1.9V8.92c0-1.88 1.14-2.9 2.8-2.9.8 0 1.49.06 1.69.08v1.97h-1.15c-.91 0-1.1.43-1.1 1.07v1.4h2.17l-.28 2.2h-1.88v5.52a7.77 7.77 0 0 0 6.7-7.71"});const s=function(e){return o.createElement("svg",i({width:21,height:21,viewBox:"0 0 21 21"},e),c)};function u(){return(u=Object.assign||function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e}).apply(this,arguments)}var d=o.createElement("path",{d:"M19.75 12.04c0-4.3-3.47-7.79-7.75-7.79a7.77 7.77 0 0 0-5.9 12.84 7.77 7.77 0 0 0 4.69 2.63v-5.49h-1.9v-2.2h1.9v-1.62c0-1.88 1.14-2.9 2.8-2.9.8 0 1.49.06 1.69.08v1.97h-1.15c-.91 0-1.1.43-1.1 1.07v1.4h2.17l-.28 2.2h-1.88v5.52a7.77 7.77 0 0 0 6.7-7.71",fill:"#A8A8A8"});const m=function(e){return o.createElement("svg",u({width:24,height:24,viewBox:"0 0 24 24",fill:"none"},e),d)};var p=n(78011);function V(){return(V=Object.assign||function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e}).apply(this,arguments)}var f=o.createElement("path",{d:"M3 4.07C3 3.48 3.5 3 4.1 3h12.8c.6 0 1.1.48 1.1 1.07v12.86c0 .59-.5 1.07-1.1 1.07H4.1A1.1 1.1 0 0 1 3 16.93V4.07z"}),v=o.createElement("path",{fillRule:"evenodd",clipRule:"evenodd",d:"M7.55 15.56V8.78H5.28v6.78h2.27zM6.4 7.86c.8 0 1.29-.52 1.29-1.17-.02-.67-.5-1.17-1.27-1.17-.78 0-1.28.5-1.28 1.17 0 .65.49 1.17 1.25 1.17h.01zM8.8 15.56h2.27v-3.79a1.24 1.24 0 0 1 1.24-1.37c.81 0 1.14.62 1.14 1.53v3.63h2.27v-3.89c0-2.08-1.12-3.05-2.61-3.05-1.22 0-1.76.68-2.06 1.15h.02v-.99H8.8c.03.64 0 6.78 0 6.78z",fill:"#fff"});const h=function(e){return o.createElement("svg",V({width:21,height:21,viewBox:"0 0 21 21",fill:"none"},e),f,v)};function g(){return(g=Object.assign||function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e}).apply(this,arguments)}var E=o.createElement("path",{d:"M19.75 5.39v13.22a1.14 1.14 0 0 1-1.14 1.14H5.39a1.14 1.14 0 0 1-1.14-1.14V5.39a1.14 1.14 0 0 1 1.14-1.14h13.22a1.14 1.14 0 0 1 1.14 1.14zM8.81 10.18H6.53v7.3H8.8v-7.3zM9 7.67a1.31 1.31 0 0 0-1.3-1.32h-.04a1.32 1.32 0 0 0 0 2.64A1.31 1.31 0 0 0 9 7.71v-.04zm8.46 5.37c0-2.2-1.4-3.05-2.78-3.05a2.6 2.6 0 0 0-2.3 1.18h-.07v-1h-2.14v7.3h2.28V13.6a1.51 1.51 0 0 1 1.36-1.63h.09c.72 0 1.26.45 1.26 1.6v3.91h2.28l.02-4.43z",fill:"#A8A8A8"});const x=function(e){return o.createElement("svg",g({width:24,height:24,viewBox:"0 0 24 24",fill:"none"},e),E)};var y=n(68455);function b(){return(b=Object.assign||function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e}).apply(this,arguments)}var w=o.createElement("path",{d:"M20 5.34c-.67.41-1.4.7-2.18.87a3.45 3.45 0 0 0-5.02-.1 3.49 3.49 0 0 0-1.02 2.47c0 .28.03.54.07.8a9.91 9.91 0 0 1-7.17-3.66 3.9 3.9 0 0 0-.5 1.74 3.6 3.6 0 0 0 1.56 2.92 3.36 3.36 0 0 1-1.55-.44V10c0 1.67 1.2 3.08 2.8 3.42-.3.06-.6.1-.94.12l-.62-.06a3.5 3.5 0 0 0 3.24 2.43 7.34 7.34 0 0 1-4.36 1.49l-.81-.05a9.96 9.96 0 0 0 5.36 1.56c6.4 0 9.91-5.32 9.9-9.9v-.5c.69-.49 1.28-1.1 1.74-1.81-.63.3-1.3.48-2 .56A3.33 3.33 0 0 0 20 5.33",fill:"#A8A8A8"});const I=function(e){return o.createElement("svg",b({width:24,height:24,viewBox:"0 0 24 24",fill:"none"},e),w)};var A=n(61600),S=n(25735),k={marginRight:"8px"},T={fill:"#3b5998"},R={fill:"#38a1f3"},O={fill:"#292929"},L=function(e){return{fill:e.colorTokens.foreground.neutral.secondary.base}},B=function(e){var t=e.buttonStyle,n=e.socialPlatform,r=(0,S.d)("enable_maim_the_meter").value,a=(0,l.I)();switch(t){case"LINK_INLINE_SHORT_LABEL":switch(n){case"FACEBOOK":return o.createElement(m,{className:a(L)});case"TWITTER":return o.createElement(I,{className:a(L)});case"LINKEDIN":return o.createElement(x,{className:a(L)});default:return null}case"BUTTON_BRANDED":switch(n){case"FACEBOOK":return r?o.createElement(p.Z,null):o.createElement(s,{className:a([T,k])});case"TWITTER":return r?o.createElement(A.Z,null):o.createElement(y.Z,{className:a([R,k])});case"LINKEDIN":return o.createElement(h,{className:a([O,k])});default:return null}default:return null}},N=n(93310),_=n(77355),P=n(47230),C=function(e){return{display:"inline-flex",alignItems:"center",":hover path":{fill:e.colorTokens.foreground.neutral.primary.base}}},D={FACEBOOK:"Facebook",TWITTER:"Twitter",LINKEDIN:"LinkedIn"};function F(e){var t=e.buttonStyle,n=e.socialPlatform,r=e.baseOnClick,l=e.href,i=(0,S.d)("enable_maim_the_meter").value,c=(0,o.useMemo)((function(){return{"aria-label":"Share on ".concat(n.toLowerCase()),onClick:function(){r();var e=Math.max((window.outerHeight||200)/2-560,100),t=(window.outerWidth||200)/2-250;window.open(l,"Social Share Window","resizable,scrollbars,status,top=".concat(e,",left=").concat(t,",height=").concat(650,",width=").concat(650))}}}),[r,l,n]),s=D[n];if(!s)return null;switch(t){case"LINK_INLINE_SHORT_LABEL":return o.createElement(N.r,a()({},c,{rules:C}),o.createElement(B,{buttonStyle:t,socialPlatform:n}),o.createElement(_.x,{display:"inline",marginLeft:"8px"},"Share on ",s));case"BUTTON_BRANDED":return o.createElement(P.zx,a()({},c,{buttonStyle:"SOCIAL",size:i?"LARGE":"REGULAR",width:i?P.nt:P.hU,padding:i?P.lF:void 0}),o.createElement(_.x,{display:"flex",alignItems:"center",justifyContent:i?"space-between":"center"},o.createElement(B,{buttonStyle:t,socialPlatform:n}),"Share with ",s,i&&o.createElement(_.x,{height:"24px",width:"24px"})));default:return null}}},85805:(e,t,n)=>{"use strict";n.d(t,{A:()=>h});var r=n(63038),a=n.n(r),o=n(67294),l=n(38352),i=n(73917),c=n(38356),s=n(18627),u=n(66411),d=n(31889);function m(){return(m=Object.assign||function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e}).apply(this,arguments)}var p=o.createElement("path",{fillRule:"evenodd",clipRule:"evenodd",d:"M15.22 4.93a.42.42 0 0 1-.12.13h.01a.45.45 0 0 1-.29.08.52.52 0 0 1-.3-.13L12.5 3v7.07a.5.5 0 0 1-.5.5.5.5 0 0 1-.5-.5V3.02l-2 2a.45.45 0 0 1-.57.04h-.02a.4.4 0 0 1-.16-.3.4.4 0 0 1 .1-.32l2.8-2.8a.5.5 0 0 1 .7 0l2.8 2.8a.42.42 0 0 1 .07.5zm-.1.14zm.88 2h1.5a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-11a2 2 0 0 1-2-2v-10a2 2 0 0 1 2-2H8a.5.5 0 0 1 .35.14c.1.1.15.22.15.35a.5.5 0 0 1-.15.35.5.5 0 0 1-.35.15H6.4c-.5 0-.9.4-.9.9v10.2a.9.9 0 0 0 .9.9h11.2c.5 0 .9-.4.9-.9V8.96c0-.5-.4-.9-.9-.9H16a.5.5 0 0 1 0-1z",fill:"currentColor"});const V=function(e){return o.createElement("svg",m({width:24,height:24,viewBox:"0 0 24 24",fill:"none"},e),p)};var f=n(68894),v=function(e){var t=e.children,n=e.source;return n?o.createElement(u.cW,{source:n},t):o.createElement(o.Fragment,null,t)},h=function(e){var t,n=e.url,r=e.title,m=e.source,p=e.ariaId,h=e.children,g=e.ariaLabel,E=e.tooltipText,x=void 0===E?"":E,y=e.postId,b=e.listId,w=e.isResponsive,I=(0,d.F)(),A=(0,f.O)(!1),S=a()(A,3),k=S[0],T=S[1],R=S[2],O=(null==I||null===(t=I.breakpoints)||void 0===t?void 0:t.md)||728,L=(0,s.Av)(),B=(0,u.f0)(m);return o.createElement(v,{source:m},o.createElement(i.J,{ariaId:p,isVisible:k,hide:R,popoverRenderFn:function(){return o.createElement(l.mX,null,h(R))}},o.createElement(c.u,{ariaControls:p,icon:o.createElement(V,null),tooltipText:x||"Share",ariaExpanded:k?"true":"false",ariaLabel:g,text:w?"Share":void 0,onClick:function(){var e,t=null===(e=window)||void 0===e?void 0:e.innerWidth;if(L.event("shareLinkPopover.clicked",{postId:y,listId:b,source:B}),n&&t&&t<O){var a={url:n,text:r||"",title:r||""};if(navigator.canShare&&navigator.canShare(a))return void navigator.share(a)}T()}})))}},37597:(e,t,n)=>{"use strict";n.d(t,{f:()=>c});var r=n(67294),a=n(99950),o=n(18627),l=n(66411),i=n(92661),c=function(e){var t,n=e.postId,c=e.socialPlatform,s=e.buttonStyle,u=(0,o.Av)(),d=(0,l.Qi)(),m=(0,i.H2)(),p=r.useCallback((function(){d&&u.event("post.shareOpen",{postId:n,source:d,dest:c.toLowerCase(),dialogType:"native"})}),[d,u,n,c]);if("FACEBOOK"===c)t=m("RedirectShowPostShare",{postId:n,channel:"facebook"});else if("TWITTER"===c)t=m("RedirectShowPostShare",{postId:n,channel:"twitter"});else{if("LINKEDIN"!==c)return null;t=m("RedirectShowPostShare",{postId:n,channel:"linkedIn"})}return r.createElement(a.T,{baseOnClick:p,href:t,socialPlatform:c,buttonStyle:s})}},28451:(e,t,n)=>{"use strict";var r;n.d(t,{j:()=>r}),function(e){e.LOHighlightCTA="lo_highlight_cta",e.LIHighlightCTA="li_highlight_cta",e.RegWall="regwall",e.SyndicatedRegwall="syndicated_regwall"}(r||(r={}))},17417:(e,t,n)=>{"use strict";n.d(t,{A:()=>d});var r=n(67154),a=n.n(r),o=n(67294),l=n(93310),i=n(77355),c=n(73917),s=n(87691),u=n(14646),d=function(e){var t=e.boxProps,n=void 0===t?{}:t,r=e.children,d=e.dismiss,m=e.dismissText,p=void 0===m?"Ok, got it.":m,V=e.isVisible,f=e.onShow,v=e.message,h=e.withBoldNew,g=void 0!==h&&h,E=e.targetDistance,x=void 0===E?5:E,y=(0,u.I)();return o.useEffect((function(){V&&f&&f()}),[V]),o.createElement(c.J,{hide:d,isVisible:V,noPortal:!0,darkTheme:!0,popoverRenderFn:function(){return o.createElement(i.x,a()({padding:"16px"},n),o.createElement(i.x,{marginBottom:"8px"},o.createElement(s.F,{scale:"S",color:"WHITE",tag:"span"},g&&o.createElement("strong",null,"New! "),v)),o.createElement(s.F,{scale:"S"},o.createElement(l.r,{onClick:d},o.createElement("span",{className:y(c.u)},p))))},referenceWidth:"100%",targetDistance:x},r)}},39944:(e,t,n)=>{"use strict";n.d(t,{M:()=>d});var r=n(59713),a=n.n(r),o=n(67294),l=n(18627),i=n(66411),c=n(18122);function s(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function u(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?s(Object(n),!0).forEach((function(t){a()(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):s(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function d(e){var t=e.source,n=e.dimension,r=e.sourceProviderData,a=void 0===r?{}:r,s=e.extendSource,d=void 0!==s&&s,m=e.children,p=e.eventData,V=void 0===p?{}:p,f=e.tag,v=void 0===f?"div":f,h=e.style,g=v,E=(0,l.Av)(),x=u(u({},V),{},{dimension:n,locationId:n}),y=(0,c.g)({onPresentedFn:function(){return E.event("upsell.viewed",x)}}),b=u(u({},a),{},{name:t,dimension:n});return o.createElement(g,{ref:y,style:h},o.createElement(i.cW,{extendSource:d,source:b},m))}}}]);
//# sourceMappingURL=https://stats.medium.build/lite/sourcemaps/2031.7407f14c.chunk.js.map